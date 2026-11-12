#![cfg(feature = "musa")]
use apxinf_model::musa_qwen38::runner::EagerRunner;

#[test]
fn invalid_capacity_is_rejected_before_checkpoint_or_device_access() {
    for capacity in [0, usize::MAX] {
        let error = match EagerRunner::load("/nonexistent/qwen38", usize::MAX, capacity) {
            Ok(_) => panic!("invalid capacity accepted"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("capacity"), "{error}");
    }
}

#[test]
#[ignore = "requires official HF BF16 checkpoint and exclusive S5000; not a parity test"]
fn full_hf_runner_finite_logits_and_reset_repeatability() {
    let path = std::env::var("APXINF_QWEN38_CHECKPOINT").unwrap();
    let mut runner = EagerRunner::load(path, 0, 3).unwrap();
    assert!(runner.forward(&[], 0).is_err());
    assert!(runner.forward(&[1, 248_320], 0).is_err());
    assert!(runner.forward(&[1], 1).is_err());
    assert!(runner.forward(&[1, 2, 3, 4], 0).is_err());
    assert!(runner.forward(&[1], usize::MAX).is_err());
    assert_eq!(runner.position(), 0);
    let first = runner.forward(&[1, 42], 0).unwrap();
    assert_eq!(first.shape().dims(), [2, 248_320]);
    let first = apxinf_musa::transfers::to_cpu(&first).unwrap();
    assert!(first
        .as_bf16()
        .unwrap()
        .iter()
        .all(|v| v.to_f32().is_finite()));
    assert_eq!(runner.position(), 2);
    assert!(runner.forward(&[1, 2], 2).is_err());
    assert_eq!(runner.position(), 2);
    runner.reset().unwrap();
    let a = runner.forward(&[1], 0).unwrap();
    let b = runner.forward(&[42], 1).unwrap();
    let a = apxinf_musa::transfers::to_cpu(&a).unwrap();
    let b = apxinf_musa::transfers::to_cpu(&b).unwrap();
    assert_eq!(&first.as_bf16().unwrap()[..248_320], a.as_bf16().unwrap());
    assert_eq!(&first.as_bf16().unwrap()[248_320..], b.as_bf16().unwrap());
    assert_eq!(runner.position(), 2);
}
