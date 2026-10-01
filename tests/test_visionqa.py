from app.visionqa import analyze_image


def test_analyze_image_function_exists():
    assert callable(analyze_image)