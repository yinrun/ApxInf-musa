//! Request position ledger for eager hybrid execution.
use apxinf_core::{Error, Result};

/// A begun token remains poisoned until its complete traversal succeeds.
/// Reset the layer caches as well before resetting this ledger.
pub struct RequestState {
    capacity: usize,
    position: usize,
    pending: bool,
}
impl RequestState {
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 { return Err(Error::Other("request capacity must be positive".into())); }
        Ok(Self { capacity, position: 0, pending: false })
    }
    pub fn position(&self) -> usize { self.position }
    pub fn begin(&mut self, position: usize) -> Result<()> {
        if self.pending || position != self.position || position >= self.capacity {
            return Err(Error::Other("request is interrupted, out of order, or exhausted; reset required after execution failure".into()));
        }
        self.pending = true;
        Ok(())
    }
    /// Call only after all layers and the output head have succeeded.
    pub fn complete(&mut self) -> Result<()> {
        if !self.pending { return Err(Error::Other("no pending token traversal".into())); }
        self.position += 1;
        self.pending = false;
        Ok(())
    }
    pub fn reset(&mut self) { self.position = 0; self.pending = false; }
}
