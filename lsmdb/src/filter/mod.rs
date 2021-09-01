pub mod bloom;

pub trait FilterPolicy: Send + Sync {
    fn name(&self) -> &str;
    fn may_contain(&self, filter: &[u8], key: &[u8]) -> bool;
    fn create_filter(&self, keys: &[Vec<u8>]) -> Vec<u8>;
}
