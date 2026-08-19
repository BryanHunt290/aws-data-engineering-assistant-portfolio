"""End-to-end, provider-neutral RAG application orchestration."""

import json
import logging
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from knowledge.application_errors import (
    ApplicationError,
    ClassificationFailure,
    LLMInvocationFailure,
    MalformedProviderFailure,
    ProviderAccessDeniedFailure,
    ProviderThrottledFailure,
    ProviderUnavailableFailure,
    PromptConstructionFailure,
    QueryEmbeddingFailure,
    RetrievalFailure,
    RoutingFailure,
)
from knowledge.application_models import (
    ApplicationRequest,
    ApplicationResponse,
    ApplicationStatus,
    EMPTY_MODEL_METADATA,
    ModelMetadata,
    RetrievalMetadata,
    SourceCitation,
)
from knowledge.config import ApplicationConfig
from knowledge.costs import CostEstimator
from knowledge.embeddings import EmbeddingProvider
from knowledge.embedding_errors import (
    EmbeddingAccessDeniedError,
    EmbeddingModelUnavailableError,
    EmbeddingThrottledError,
    MalformedEmbeddingResponseError,
)
from knowledge.intents import (
    ClassificationResult,
    Intent,
    IntentClassifier,
)
from knowledge.llm import GenerationResult, LLMProvider
from knowledge.llm_errors import (
    LLMAccessDeniedError,
    LLMModelUnavailableError,
    LLMThrottledError,
    MalformedLLMResponseError,
)
from knowledge.prompting import PromptBuilder, PromptContext
from knowledge.retrieval import RetrievalResult, Retriever
from knowledge.routing import RequestRouter, Route, RoutingPlan
from knowledge.vector_store import VectorStore


class RAGApplicationService:
    """Compose classification through generation without creating clients."""

    def __init__(
        self,
        *,
        classifier: IntentClassifier,
        router: RequestRouter,
        embedding_provider: EmbeddingProvider,
        retriever: Retriever | None,
        prompt_builder: PromptBuilder,
        llm_provider: LLMProvider,
        vector_store: VectorStore | None = None,
        config: ApplicationConfig | None = None,
        cost_estimator: CostEstimator | None = None,
        runtime_mode: str = "bedrock",
        event_logger: logging.Logger | None = None,
        monotonic_clock: Callable[[], float] = perf_counter,
    ) -> None:
        self._classifier = classifier
        self._router = router
        self._embedding_provider = embedding_provider
        self._retriever = retriever
        self._vector_store = vector_store
        if (retriever is None) == (vector_store is None):
            raise ValueError(
                "Configure exactly one retriever or scoped vector store"
            )
        self._prompt_builder = prompt_builder
        self._llm_provider = llm_provider
        self._config = config or ApplicationConfig()
        self._cost_estimator = cost_estimator
        self._runtime_mode = runtime_mode.strip().lower()
        if not self._runtime_mode:
            raise ValueError("runtime_mode cannot be empty")
        self._logger = event_logger or logging.getLogger(__name__)
        self._clock = monotonic_clock
        if prompt_builder.prompt_version != self._config.prompt_version:
            raise ValueError(
                "PromptBuilder version must match ApplicationConfig"
            )

    def handle(self, request: ApplicationRequest) -> ApplicationResponse:
        """Handle one request and return a safe terminal response."""

        started = self._clock()
        empty_retrieval = RetrievalMetadata(
            attempted=False,
            result_count=0,
            requested_top_k=None,
            minimum_similarity=None,
            context_characters=0,
        )
        try:
            request.validate(self._config)
        except Exception as error:
            return self._failure(
                request=request,
                started=started,
                error=ApplicationError(
                    "Request validation failed"
                ),
                retrieval_metadata=empty_retrieval,
            )

        conversation, conversation_truncated = (
            request.bounded_conversation(self._config)
        )
        warnings: list[str] = []
        if conversation_truncated:
            warnings.append(
                "Prior conversation was truncated to configured limits."
            )
        conversation_text = "\n".join(
            f"{message.role.value}: {message.content}"
            for message in conversation
        )

        try:
            classification = self._classifier.classify(
                request.query,
                conversation_context=conversation_text or None,
                client_id=request.client_id,
                environment=request.environment,
                metadata=request.metadata,
            )
            if not isinstance(classification, ClassificationResult):
                raise TypeError("Classifier returned an invalid result")
        except Exception:
            return self._failure(
                request=request,
                started=started,
                error=ClassificationFailure(),
                retrieval_metadata=empty_retrieval,
                warnings=warnings,
            )

        try:
            plan = self._router.route(
                classification,
                client_id=request.client_id,
                environment=request.environment,
            )
            if not isinstance(plan, RoutingPlan):
                raise TypeError("Router returned an invalid plan")
        except Exception:
            return self._failure(
                request=request,
                started=started,
                error=RoutingFailure(),
                retrieval_metadata=empty_retrieval,
                classification=classification,
                warnings=warnings,
            )

        if plan.selected_route == Route.REJECTION_OR_SAFETY_REVIEW:
            return self._terminal_without_model(
                request=request,
                plan=plan,
                started=started,
                status=ApplicationStatus.SAFETY_REVIEW_REQUIRED,
                answer=(
                    "No action was executed. This request requires explicit "
                    "approval, exact target confirmation, and a safety review."
                ),
                retrieval_metadata=empty_retrieval,
                warnings=warnings,
            )
        if plan.selected_route == Route.APPROVAL_REQUIRED:
            return self._terminal_without_model(
                request=request,
                plan=plan,
                started=started,
                status=ApplicationStatus.APPROVAL_REQUIRED,
                answer=(
                    "No action was executed. Explicit approval is required "
                    "before any deployment or tool invocation."
                ),
                retrieval_metadata=empty_retrieval,
                warnings=warnings,
            )
        if plan.selected_route == Route.TOOL_EXECUTION:
            warnings.append(
                "Tool execution is not implemented; no current-state claim "
                "or action was produced."
            )
            return self._terminal_without_model(
                request=request,
                plan=plan,
                started=started,
                status=ApplicationStatus.INSUFFICIENT_CONTEXT,
                answer=(
                    "This request needs a scoped tool result, but tool "
                    "execution is not enabled. No action was taken."
                ),
                retrieval_metadata=empty_retrieval,
                warnings=warnings,
            )

        contexts: tuple[PromptContext, ...] = ()
        retrieval_metadata = empty_retrieval
        if plan.retrieval_required:
            try:
                contexts, retrieval_metadata = self._retrieve(
                    request,
                    plan,
                    conversation_characters=sum(
                        len(message.content) for message in conversation
                    ),
                )
            except QueryEmbeddingFailure as error:
                return self._failure(
                    request=request,
                    started=started,
                    error=self._provider_failure(
                        error.__cause__,
                        error,
                    ),
                    retrieval_metadata=RetrievalMetadata(
                        attempted=True,
                        result_count=0,
                        requested_top_k=self._retrieval_limit(plan),
                        minimum_similarity=self._config.minimum_similarity,
                        context_characters=0,
                    ),
                    classification=classification,
                    plan=plan,
                    warnings=warnings,
                )
            except RetrievalFailure as error:
                return self._failure(
                    request=request,
                    started=started,
                    error=error,
                    retrieval_metadata=RetrievalMetadata(
                        attempted=True,
                        result_count=0,
                        requested_top_k=self._retrieval_limit(plan),
                        minimum_similarity=self._config.minimum_similarity,
                        context_characters=0,
                    ),
                    classification=classification,
                    plan=plan,
                    warnings=warnings,
                )
            except Exception:
                return self._failure(
                    request=request,
                    started=started,
                    error=RetrievalFailure(),
                    retrieval_metadata=RetrievalMetadata(
                        attempted=True,
                        result_count=0,
                        requested_top_k=self._retrieval_limit(plan),
                        minimum_similarity=self._config.minimum_similarity,
                        context_characters=0,
                    ),
                    classification=classification,
                    plan=plan,
                    warnings=warnings,
                )
            if not contexts:
                warnings.append(
                    "No scoped retrieval result passed the configured "
                    "similarity and context limits."
                )
                return self._insufficient_context(
                    request=request,
                    plan=plan,
                    started=started,
                    retrieval_metadata=retrieval_metadata,
                    sources=(),
                    warnings=warnings,
                )

        try:
            built_prompt = self._prompt_builder.build(
                request=request,
                conversation=conversation,
                routing_plan=plan,
                contexts=contexts,
            )
        except Exception as error:
            return self._failure(
                request=request,
                started=started,
                error=PromptConstructionFailure(),
                retrieval_metadata=retrieval_metadata,
                classification=classification,
                plan=plan,
                sources=tuple(
                    context.citation for context in contexts
                ),
                warnings=warnings,
            )

        try:
            generation = self._llm_provider.generate(
                system_prompt=built_prompt.system_prompt,
                user_prompt=built_prompt.user_prompt,
                model_parameters={
                    "temperature": self._config.temperature,
                    "maximum_tokens": self._config.maximum_tokens,
                },
            )
            if not isinstance(generation, GenerationResult):
                raise TypeError("LLM provider returned an invalid result")
            if (
                not generation.generated_text.strip()
                and not generation.indicates_insufficient_context
            ):
                raise ValueError("LLM provider returned empty text")
        except Exception as error:
            return self._failure(
                request=request,
                started=started,
                error=self._provider_failure(
                    error,
                    LLMInvocationFailure(),
                ),
                retrieval_metadata=retrieval_metadata,
                classification=classification,
                plan=plan,
                sources=tuple(
                    context.citation for context in contexts
                ),
                warnings=warnings,
            )

        model_metadata = self._model_metadata(generation, request)
        sources = tuple(context.citation for context in contexts)
        if generation.indicates_insufficient_context:
            warnings.append(
                "The language model explicitly reported insufficient context."
            )
            return self._insufficient_context(
                request=request,
                plan=plan,
                started=started,
                retrieval_metadata=retrieval_metadata,
                sources=sources,
                warnings=warnings,
                model_metadata=model_metadata,
            )

        response = ApplicationResponse(
            request_id=request.request_id,
            answer=generation.generated_text.strip(),
            intent=plan.intent,
            route=plan.selected_route,
            confidence=plan.classifier_confidence,
            sources=sources,
            retrieval_metadata=retrieval_metadata,
            model_metadata=model_metadata,
            approval_required=plan.approval_required,
            safety_review_required=plan.safety_review_required,
            latency_ms=self._elapsed_ms(started),
            warnings=tuple(warnings),
            status=ApplicationStatus.COMPLETED,
        )
        self._emit(request, response)
        return response

    def _retrieve(
        self,
        request: ApplicationRequest,
        plan: RoutingPlan,
        *,
        conversation_characters: int,
    ) -> tuple[tuple[PromptContext, ...], RetrievalMetadata]:
        try:
            vectors = self._embedding_provider.embed([request.query])
            if len(vectors) != 1 or not vectors[0]:
                raise ValueError("Embedding provider returned invalid output")
            query_vector = vectors[0]
        except Exception as error:
            raise QueryEmbeddingFailure() from error

        limit = self._retrieval_limit(plan)
        try:
            if self._vector_store is not None:
                results = self._vector_store.retrieve(
                    query_vector,
                    client_id=request.client_id,
                    environment=request.environment,
                    filters=self._retrieval_filters(request),
                    top_k=limit,
                    minimum_similarity=self._config.minimum_similarity,
                )
            else:
                if self._retriever is None:  # pragma: no cover - constructor
                    raise RuntimeError("Retriever is not configured")
                results = self._retriever.retrieve(
                    query_vector,
                    top_k=limit,
                    minimum_similarity=self._config.minimum_similarity,
                )
        except Exception as error:
            raise RetrievalFailure() from error

        contexts: list[PromptContext] = []
        seen: set[tuple[str, str]] = set()
        filtered_for_scope = 0
        deduplicated = 0
        remaining_characters = max(
            0,
            self._config.context_length_limit
            - conversation_characters,
        )
        for result in results:
            if not isinstance(result, RetrievalResult):
                raise RetrievalFailure(
                    "Retriever returned an invalid result"
                )
            if result.similarity_score < self._config.minimum_similarity:
                continue
            if not self._matches_scope(result, request):
                filtered_for_scope += 1
                continue
            key = (result.document_id, result.chunk_id)
            if key in seen:
                deduplicated += 1
                continue
            seen.add(key)
            if remaining_characters <= 0:
                break
            text = result.text[:remaining_characters].strip()
            if not text:
                continue
            metadata = dict(result.metadata)
            source_id = f"S{len(contexts) + 1}"
            citation = SourceCitation(
                source_id=source_id,
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                source_name=result.source,
                object_key=str(
                    metadata.get("object_key")
                    or metadata.get("source_object_key")
                    or result.source
                ),
                similarity_score=result.similarity_score,
                page=metadata.get("page"),
                section=(
                    str(metadata["section"])
                    if metadata.get("section") is not None
                    else None
                ),
                metadata=metadata,
            )
            contexts.append(PromptContext(citation=citation, text=text))
            remaining_characters -= len(text)
            if len(contexts) >= limit:
                break

        context_characters = sum(
            len(context.text) for context in contexts
        )
        return tuple(contexts), RetrievalMetadata(
            attempted=True,
            result_count=len(contexts),
            requested_top_k=limit,
            minimum_similarity=self._config.minimum_similarity,
            context_characters=context_characters,
            filtered_for_scope=filtered_for_scope,
            deduplicated=deduplicated,
        )

    @staticmethod
    def _matches_scope(
        result: RetrievalResult,
        request: ApplicationRequest,
    ) -> bool:
        metadata = result.metadata
        return (
            metadata.get("client_id") == request.client_id
            and metadata.get("environment") == request.environment
        )

    @staticmethod
    def _retrieval_filters(request: ApplicationRequest) -> dict[str, Any]:
        supported = {
            "agent",
            "document_type",
            "knowledge_domain",
            "knowledge_namespace",
            "source",
        }
        return {
            key: value
            for key, value in request.metadata.items()
            if key in supported and value is not None
        }

    def _retrieval_limit(self, plan: RoutingPlan) -> int:
        return min(
            self._config.maximum_retrieved_chunks,
            plan.retrieval_top_k
            or self._config.maximum_retrieved_chunks,
        )

    def _terminal_without_model(
        self,
        *,
        request: ApplicationRequest,
        plan: RoutingPlan,
        started: float,
        status: ApplicationStatus,
        answer: str,
        retrieval_metadata: RetrievalMetadata,
        warnings: Sequence[str],
    ) -> ApplicationResponse:
        response = ApplicationResponse(
            request_id=request.request_id,
            answer=answer,
            intent=plan.intent,
            route=plan.selected_route,
            confidence=plan.classifier_confidence,
            sources=(),
            retrieval_metadata=retrieval_metadata,
            model_metadata=EMPTY_MODEL_METADATA,
            approval_required=plan.approval_required,
            safety_review_required=plan.safety_review_required,
            latency_ms=self._elapsed_ms(started),
            warnings=tuple(warnings),
            status=status,
        )
        self._emit(request, response)
        return response

    def _insufficient_context(
        self,
        *,
        request: ApplicationRequest,
        plan: RoutingPlan,
        started: float,
        retrieval_metadata: RetrievalMetadata,
        sources: tuple[SourceCitation, ...],
        warnings: Sequence[str],
        model_metadata: ModelMetadata = EMPTY_MODEL_METADATA,
    ) -> ApplicationResponse:
        response = ApplicationResponse(
            request_id=request.request_id,
            answer=(
                "I do not have enough scoped, verified context to answer "
                "without guessing. Provide relevant documentation, resource "
                "details, error text, or a confirmed tool result."
            ),
            intent=plan.intent,
            route=plan.selected_route,
            confidence=plan.classifier_confidence,
            sources=sources,
            retrieval_metadata=retrieval_metadata,
            model_metadata=model_metadata,
            approval_required=plan.approval_required,
            safety_review_required=plan.safety_review_required,
            latency_ms=self._elapsed_ms(started),
            warnings=tuple(warnings),
            status=ApplicationStatus.INSUFFICIENT_CONTEXT,
        )
        self._emit(request, response)
        return response

    def _failure(
        self,
        *,
        request: ApplicationRequest,
        started: float,
        error: ApplicationError,
        retrieval_metadata: RetrievalMetadata,
        classification: ClassificationResult | None = None,
        plan: RoutingPlan | None = None,
        sources: tuple[SourceCitation, ...] = (),
        warnings: Sequence[str] = (),
    ) -> ApplicationResponse:
        response = ApplicationResponse(
            request_id=request.request_id,
            answer=error.user_message,
            intent=(
                plan.intent
                if plan is not None
                else (
                    classification.intent
                    if classification is not None
                    else Intent.UNKNOWN
                )
            ),
            route=(
                plan.selected_route
                if plan is not None
                else Route.REQUIREMENTS_GATHERING
            ),
            confidence=(
                plan.classifier_confidence
                if plan is not None
                else (
                    classification.confidence
                    if classification is not None
                    else 0.0
                )
            ),
            sources=sources,
            retrieval_metadata=retrieval_metadata,
            model_metadata=EMPTY_MODEL_METADATA,
            approval_required=(
                plan.approval_required if plan is not None else False
            ),
            safety_review_required=(
                plan.safety_review_required if plan is not None else False
            ),
            latency_ms=self._elapsed_ms(started),
            warnings=tuple(warnings),
            status=ApplicationStatus.FAILED,
            error_category=error.category,
        )
        self._emit(request, response)
        return response

    def _model_metadata(
        self,
        generation: GenerationResult,
        request: ApplicationRequest,
    ) -> ModelMetadata:
        cost_estimate = None
        if self._cost_estimator is not None:
            metadata = generation.provider_metadata
            cost_estimate = self._cost_estimator.estimate(
                model_id=generation.model_id,
                input_token_count=generation.input_token_count,
                output_token_count=generation.output_token_count,
                cache_read_token_count=metadata.get(
                    "cache_read_token_count"
                ),
                cache_write_token_count=metadata.get(
                    "cache_write_token_count"
                ),
                region=self._config.bedrock_llm_region,
                runtime_mode=self._runtime_mode,
            )
        return ModelMetadata(
            provider_name=self._llm_provider.provider_name,
            model_id=generation.model_id,
            input_token_count=generation.input_token_count,
            output_token_count=generation.output_token_count,
            finish_reason=generation.finish_reason,
            latency_ms=generation.latency_ms,
            provider_metadata=generation.provider_metadata,
            cost_estimate=cost_estimate,
        )

    @staticmethod
    def _provider_failure(
        error: BaseException | None,
        fallback: ApplicationError,
    ) -> ApplicationError:
        if isinstance(
            error,
            (LLMThrottledError, EmbeddingThrottledError),
        ):
            return ProviderThrottledFailure()
        if isinstance(
            error,
            (LLMAccessDeniedError, EmbeddingAccessDeniedError),
        ):
            return ProviderAccessDeniedFailure()
        if isinstance(
            error,
            (
                LLMModelUnavailableError,
                EmbeddingModelUnavailableError,
            ),
        ):
            return ProviderUnavailableFailure()
        if isinstance(
            error,
            (
                MalformedLLMResponseError,
                MalformedEmbeddingResponseError,
            ),
        ):
            return MalformedProviderFailure()
        return fallback

    def _elapsed_ms(self, started: float) -> float:
        return max(0.0, (self._clock() - started) * 1_000)

    def _emit(
        self,
        request: ApplicationRequest,
        response: ApplicationResponse,
    ) -> None:
        event: dict[str, Any] = {
            "application_version": self._config.application_version,
            "client_id": request.client_id,
            "elapsed_ms": round(response.latency_ms, 3),
            "environment": request.environment,
            "error_category": response.error_category,
            "event": "rag_application_request",
            "intent": response.intent.value,
            "model_id": response.model_metadata.model_id,
            "request_id": request.request_id,
            "retrieval_result_count": (
                response.retrieval_metadata.result_count
            ),
            "route": response.route.value,
            "status": response.status.value,
        }
        estimate = response.model_metadata.cost_estimate
        event.update(
            {
                "input_token_count": (
                    response.model_metadata.input_token_count
                ),
                "output_token_count": (
                    response.model_metadata.output_token_count
                ),
                "pricing_version": (
                    estimate.pricing_version if estimate else None
                ),
                "estimated_total_cost": (
                    str(estimate.total_estimated_cost)
                    if estimate
                    and estimate.total_estimated_cost is not None
                    else None
                ),
                "currency": estimate.currency if estimate else None,
            }
        )
        self._logger.info(json.dumps(event, sort_keys=True))
