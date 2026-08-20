use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

/// Wraps the `Result<T, String>` shape every service function already
/// returns into an HTTP response: `Ok` serializes as `200 application/json`,
/// `Err` becomes `500` with `{"error": ...}`.
pub struct ApiError(String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": self.0 }))).into_response()
    }
}

impl From<String> for ApiError {
    fn from(value: String) -> Self {
        ApiError(value)
    }
}

pub type ApiResult<T> = Result<Json<T>, ApiError>;
