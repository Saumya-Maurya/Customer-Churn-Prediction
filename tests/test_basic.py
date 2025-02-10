def test_import():
    """Test that we can import our modules."""
    try:
        from src.models import predict
        assert True
    except ImportError:
        assert False, "Failed to import prediction module"

def test_directories():
    """Test that required directories exist."""
    import os
    assert os.path.exists('data'), "Data directory not found"
    assert os.path.exists('models'), "Models directory not found" 