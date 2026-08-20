    use super::*;
    #[test]
    fn validates_dimensions_and_values() {
        assert!(validate_embedding(&[0.1, 0.2], 2).is_ok());
        assert!(validate_embedding(&[0.1], 2).is_err());
        assert!(validate_embedding(&[f32::NAN], 1).is_err());
    }
