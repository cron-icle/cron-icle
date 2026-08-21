//! Durable storage: the SQLite event database and the batching writer that
//! keyboard/mouse hooks hand normalized events to off their hot path.

pub mod sqlite;
pub mod writer;
