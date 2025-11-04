# Contributing to Traffic Video Analyzer

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

- Be respectful and considerate
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Keep discussions professional

## How to Contribute

### Reporting Bugs

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Detailed steps to reproduce the issue
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**:
   - OS and version
   - Python version
   - Relevant package versions
   - GPU information (if applicable)
6. **Screenshots/Logs**: If applicable

### Suggesting Features

Feature suggestions are welcome! Please:

1. Check if the feature already exists or is planned
2. Provide a clear description of the feature
3. Explain the use case and benefits
4. Suggest implementation approach (optional)

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**:
   - Follow the coding style
   - Add comments for complex logic
   - Update documentation if needed
   - Add tests if applicable
4. **Test your changes**:
   - Test on clean environment
   - Verify existing functionality still works
   - Test edge cases
5. **Commit your changes**:
   ```bash
   git commit -m "Add: descriptive commit message"
   ```
   Use clear, descriptive commit messages.
6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request**:
   - Provide clear description
   - Reference related issues
   - Add screenshots if UI changes

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/traffic-video-analyzer.git
   cd traffic-video-analyzer
   ```

2. Create virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Install development dependencies:
   ```bash
   pip install pytest black flake8 mypy
   ```

## Coding Standards

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use descriptive variable and function names

### Code Formatting

We recommend using `black`:
```bash
black *.py
```

### Type Hints

Add type hints where helpful:
```python
def process_frame(frame: np.ndarray) -> dict:
    ...
```

### Documentation

- Add docstrings to functions and classes
- Use Google or NumPy docstring style
- Document complex algorithms
- Keep comments up to date

### Example Docstring

```python
def detect_vehicles(frame: np.ndarray, model: YOLO) -> list:
    """Detect vehicles in a video frame.
    
    Args:
        frame: Input BGR frame as numpy array
        model: YOLOv8 model instance
        
    Returns:
        List of detections with bounding boxes and confidence scores
        
    Raises:
        ValueError: If frame is empty or invalid
    """
    ...
```

## Testing

### Writing Tests

- Write tests for new features
- Test edge cases and error handling
- Use descriptive test names
- Keep tests simple and focused

### Running Tests

```bash
pytest tests/
```

## Project Structure

- `vehicle_detection.py`: Main application
- `mqtt.py`: MQTT-enabled version
- `training.py`: Model training
- `examples/`: Example files and configs
- `docs/`: Documentation
- `tests/`: Test suite (to be added)

## Commit Message Guidelines

Use clear, descriptive commit messages:

- **Format**: `Type: Description`
- **Types**: `Add`, `Fix`, `Update`, `Remove`, `Refactor`, `Docs`
- **Example**: `Add: Support for RTSP streaming`

## Review Process

1. All PRs require review
2. Address review comments promptly
3. Be open to feedback and suggestions
4. Keep PRs focused and reasonably sized

## Questions?

- Open an issue for questions
- Check existing issues first
- Be patient - maintainers are volunteers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing! 🎉
