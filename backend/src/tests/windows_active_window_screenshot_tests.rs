    use super::*;
    #[test]
    fn png_encoder_emits_valid_signature() {
        let png = encode_png_rgba(1, 1, &[255, 0, 0, 255]).unwrap();
        assert_eq!(&png[..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
        assert!(png.windows(4).any(|chunk| chunk == b"IHDR"));
    }
