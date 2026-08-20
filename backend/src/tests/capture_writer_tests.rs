    use super::*;

    #[test]
    fn evicts_oldest_entry_past_capacity() {
        let mut cache = ScreenshotCache::new(2);
        cache.insert("a".into(), vec![1]);
        cache.insert("b".into(), vec![2]);
        cache.insert("c".into(), vec![3]);
        assert!(cache.take("a").is_none());
        assert_eq!(cache.take("b"), Some(vec![2]));
        assert_eq!(cache.take("c"), Some(vec![3]));
    }

    #[test]
    fn take_removes_the_entry() {
        let mut cache = ScreenshotCache::new(4);
        cache.insert("x".into(), vec![9]);
        assert_eq!(cache.take("x"), Some(vec![9]));
        assert!(cache.take("x").is_none());
    }
