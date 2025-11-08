/// Extension trait for std::fs::Metadata to extract timestamps as i64.
use std::fs::Metadata;

pub trait MetadataExt {
    /// Extract modified time as seconds since UNIX_EPOCH, or None if unavailable.
    fn mtime_as_secs(&self) -> Option<i64>;

    /// Extract accessed time as seconds since UNIX_EPOCH, or None if unavailable.
    fn atime_as_secs(&self) -> Option<i64>;
}

impl MetadataExt for Metadata {
    fn mtime_as_secs(&self) -> Option<i64> {
        self.modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
    }

    fn atime_as_secs(&self) -> Option<i64> {
        self.accessed()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
    }
}
