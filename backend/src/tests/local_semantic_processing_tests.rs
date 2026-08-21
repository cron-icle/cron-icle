    use super::*;
    #[test]
    fn accepts_valid_output() {
        assert!(validate_model_output(SemanticModelOutput {
            category: "work".into(),
            summary: "Edited a file".into(),
            entities: vec![],
            relationships: vec![],
            confidence: 0.8
        })
        .is_ok());
    }
    #[test]
    fn rejects_invalid_confidence() {
        assert!(validate_model_output(SemanticModelOutput {
            category: "work".into(),
            summary: "summary".into(),
            entities: vec![],
            relationships: vec![],
            confidence: 2.0
        })
        .is_err());
    }
    #[test]
    fn parses_structured_model_json() {
        let output = parse_and_validate_model_json(
            r#"{"category":"work","summary":"Edited a file","confidence":0.7}"#,
        )
        .unwrap();
        assert_eq!(output.category, "work");
    }
    #[test]
    fn rejects_oversized_model_json() {
        assert!(parse_and_validate_model_json(&"x".repeat(65 * 1024)).is_err());
    }
    #[test]
    fn validates_supported_image_inputs() {
        assert!(validate_image_input(&[137, 80, 78, 71, 13, 10, 26, 10, 1]).is_ok());
        assert!(validate_image_input(&[1, 2, 3]).is_err());
    }
