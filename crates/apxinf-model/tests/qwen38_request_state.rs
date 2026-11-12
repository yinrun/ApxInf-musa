#![cfg(feature="musa")]
use apxinf_model::musa_qwen38::request::RequestState;
#[test]
fn request_failure_requires_reset_and_success_advances_position() {
 let mut state=RequestState::new(2).unwrap();
 assert!(state.begin(1).is_err());
 state.begin(0).unwrap();
 // No completion: a failed or interrupted traversal must not be retried.
 assert!(state.begin(0).is_err());
 state.reset();state.begin(0).unwrap();state.complete().unwrap();
 assert_eq!(state.position(),1);
 assert!(state.begin(0).is_err());
 state.begin(1).unwrap();state.complete().unwrap();
 assert!(state.begin(2).is_err());
 state.reset();assert_eq!(state.position(),0);
 assert!(RequestState::new(0).is_err());
}
