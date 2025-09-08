# fetch-my-weather LLM Guide

This guide provides comprehensive information about the `fetch-my-weather` Python package for large language models (LLMs) to understand its functionality, architecture, and usage patterns. 

## Package Overview

`fetch-my-weather` is a beginner-friendly Python package for fetching weather data, designed primarily for educational purposes. The package acts as a wrapper around the [wttr.in](https://github.com/chubin/wttr.in) weather service, providing an easy-to-use API with built-in error handling and caching.

### Core Features

- Easy access to weather data from wttr.in
- Moon phase information
- Location-based weather (cities, airports, coordinates)
- Multiple language support
- Text, JSON, and PNG output formats
- Built-in caching to reduce API requests
- Beginner-friendly error handling (no exceptions)
- Mock data mode for development and testing
- Pydantic models for type-safe JSON data handling
- Designed for teaching Python and API interactions

## Installation

The package can be installed via pip:

```python
pip install fetch-my-weather
```

## API Reference

### Main Function

The primary function users will interact with is `get_weather()`:

```python
fetch_my_weather.get_weather(
    location: str = "",
    units: str = "",
    view_options: str = "",
    lang: Optional[str] = None,
    is_png: bool = False,
    png_options: str = "",
    is_moon: bool = False,
    moon_date: Optional[str] = None,
    moon_location_hint: Optional[str] = None,
    format: Literal["text", "json", "raw_json", "png"] = "json",
    use_mock: Optional[bool] = None,
    with_metadata: bool = False,
) -> Union[str, bytes, Dict[str, Any], WeatherResponse, ResponseWrapper]
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `location` | str | Location identifier (city name, airport code, coordinates, etc.). Empty for current location. |
| `units` | str | Units system: `m` (metric, default), `u` (US/imperial), `M` (wind in m/s) |
| `view_options` | str | Display options: `0`-`3` (forecast days), `n` (narrow), `q` (quiet), etc. |
| `lang` | str | Language code (e.g., `en`, `fr`, `es`, `ru`, `zh-cn`) |
| `is_png` | bool | Deprecated: If `True`, return PNG image as bytes instead of text. Use `format="png"` instead. |
| `png_options` | str | PNG-specific options: `p` (padding), `t` (transparency), etc. |
| `is_moon` | bool | If `True`, show moon phase instead of weather |
| `moon_date` | str | Date for moon phase in `YYYY-MM-DD` format (with `is_moon=True`) |
| `moon_location_hint` | str | Location hint for moon phase (e.g., `,+US`, `,+Paris`) |
| `format` | str | Output format: `"text"`, `"json"` (default, Pydantic model), `"raw_json"` (Python dict), or `"png"` |
| `use_mock` | bool | If set, overrides the global mock mode setting for this request only |
| `with_metadata` | bool | If `True`, returns a `ResponseWrapper` containing both the response data and metadata about the response |

#### Return Value

- If `with_metadata=True`: Returns a `ResponseWrapper` object containing both the response data and metadata.
- If successful with `format="text"`: Returns the weather report as a string.
- If successful with `format="png"`: Returns the PNG image data as bytes.
- If successful with `format="json"`: Returns a `WeatherResponse` Pydantic model.
- If successful with `format="raw_json"`: Returns the raw JSON data as a Python dictionary.
- If an error occurs with `with_metadata=False`: Returns an error message string (starting with "Error:").
- If an error occurs with `with_metadata=True`: Returns mock data with error information in the metadata.

### Configuration Functions

```python
# Set cache duration in seconds (0 to disable)
fetch_my_weather.set_cache_duration(seconds: int) -> int

# Clear the current cache
fetch_my_weather.clear_cache() -> int

# Set a custom user agent string
fetch_my_weather.set_user_agent(user_agent: str) -> str

# Enable or disable mock data mode globally
fetch_my_weather.set_mock_mode(enabled: bool) -> bool
```

## Usage Examples

### Basic Usage

```python
import fetch_my_weather

# Get weather for current location (based on IP)
# Returns WeatherResponse Pydantic model by default (JSON format)
weather = fetch_my_weather.get_weather()
print(weather)

# Get weather for a specific city as text
paris_weather = fetch_my_weather.get_weather(location="Paris", format="text")
print(paris_weather)

# Get weather with compact view and metric units
berlin_weather = fetch_my_weather.get_weather(
    location="Berlin", view_options="0", units="m"
)
print(berlin_weather)
```

### Using JSON Format with Pydantic Models

```python
import fetch_my_weather

# Get weather data as a Pydantic model
weather = fetch_my_weather.get_weather(location="Paris")

# Access data with type safety
if weather.current_condition:
    current = weather.current_condition[0]
    print(f"Temperature: {current.temp_C}°C / {current.temp_F}°F")
    
    if current.weatherDesc:
        print(f"Condition: {current.weatherDesc[0].value}")
        
    print(f"Humidity: {current.humidity}%")
    print(f"Wind: {current.windspeedKmph} km/h, {current.winddir16Point}")
```

### Using Raw JSON Format

```python
import fetch_my_weather

# Get weather data as a raw Python dictionary
weather = fetch_my_weather.get_weather(location="Paris", format="raw_json")

# Access data using dictionary key/value access
if "current_condition" in weather and weather["current_condition"]:
    current = weather["current_condition"][0]
    print(f"Temperature: {current.get('temp_C')}°C / {current.get('temp_F')}°F")
    
    if "weatherDesc" in current and current["weatherDesc"]:
        print(f"Condition: {current['weatherDesc'][0]['value']}")
        
    print(f"Humidity: {current.get('humidity')}%")
    print(f"Wind: {current.get('windspeedKmph')} km/h, {current.get('winddir16Point')}")
```

### Using Mock Mode

Mock data provides realistic sample data that matches the structure of real API responses, but with clear visual indicators to help users identify when they're seeing mock data:

1. In text format, mock data includes `[MOCK DATA]` in the output
2. In JSON format, mock data includes a `mock_data_notice` field
3. The location is always displayed as "MockCity, MockLand" regardless of the location parameter
4. When using `with_metadata=True`, the `is_mock` field is set to `True`

```python
import fetch_my_weather

# Enable mock mode globally
fetch_my_weather.set_mock_mode(True)

# Get mock weather data (no API call is made)
# The location doesn't matter in mock mode, it will always return data for "MockCity"
mock_weather = fetch_my_weather.get_weather(location="AnyCity")

# You can identify mock data in various ways:
if mock_weather.current_condition:
    # Through the location name
    location = mock_weather.nearest_area[0].areaName[0].value
    country = mock_weather.nearest_area[0].country[0].value
    print(f"Location: {location}, {country}")  # Will show "MockCity, MockLand"
    
    # Through the mock_data_notice field (in raw_json format)
    if hasattr(mock_weather, "mock_data_notice"):
        print(f"Notice: {mock_weather.mock_data_notice}")
    
    print(f"Mock temperature: {mock_weather.current_condition[0].temp_C}°C")

# Override mock mode for a specific request
real_weather = fetch_my_weather.get_weather(location="Tokyo", use_mock=False)

# Disable mock mode
fetch_my_weather.set_mock_mode(False)
```

### Getting Moon Phase Data

```python
# Current moon phase
moon = fetch_my_weather.get_weather(is_moon=True)
print(moon)

# Moon phase for specific date
christmas_moon = fetch_my_weather.get_weather(is_moon=True, moon_date="2025-12-25")
print(christmas_moon)
```

### Getting PNG Weather Images

```python
# Weather as PNG (returns bytes)
city_png = fetch_my_weather.get_weather(location="Paris", format="png")

# Save PNG to file
with open("paris_weather.png", "wb") as f:
    f.write(city_png)

# PNG with transparency
transparent_png = fetch_my_weather.get_weather(
    location="Tokyo", format="png", png_options="t"
)

# Using the deprecated method (still works but not recommended)
old_method_png = fetch_my_weather.get_weather(location="Paris", is_png=True)
```

### Caching Control

```python
# Set cache duration to 30 minutes
fetch_my_weather.set_cache_duration(1800)

# Disable caching
fetch_my_weather.set_cache_duration(0)

# Clear the cache
fetch_my_weather.clear_cache()
```

### Error Handling Pattern

```python
# fetch-my-weather never raises exceptions, it returns error messages as strings
result = fetch_my_weather.get_weather(location="NonExistentPlace12345")

# Check if result is an error message
if isinstance(result, str) and result.startswith("Error:"):
    print(f"Something went wrong: {result}")
else:
    print("Weather data received successfully")
    
    # If using JSON format (default)
    if hasattr(result, "current_condition"):
        print(f"Temperature: {result.current_condition[0].temp_C}°C")
```

### Rate Limiting Fallback

When encountering a 503 status code from wttr.in (typically indicating rate limiting), the package will automatically return mock data when using JSON formats:

```python
# Even during rate limiting, this returns a valid Pydantic model instead of an error
weather = fetch_my_weather.get_weather(location="Paris")

# Access data as normal - it's valid mock data instead of an error string
if weather.current_condition:
    print(f"Temperature: {weather.current_condition[0].temp_C}°C")
```

### Response Metadata

You can use the `with_metadata=True` parameter to get information about the response's source and any errors:

```python
# Get response with metadata
response = fetch_my_weather.get_weather(
    location="Paris",
    with_metadata=True
)

# Response is now a ResponseWrapper object
if isinstance(response, ResponseWrapper):
    # Access the metadata
    metadata = response.metadata
    print(f"Real API data: {metadata.is_real_data}")
    print(f"Cached data: {metadata.is_cached}")
    print(f"Mock data: {metadata.is_mock}")
    
    # If there was an error, it's captured in the metadata
    if metadata.error_type:
        print(f"Error occurred: {metadata.error_type}")
        print(f"Error message: {metadata.error_message}")
    
    # Access the actual data (which will be mock data if there was an error)
    data = response.data
    if hasattr(data, "current_condition") and data.current_condition:
        print(f"Temperature: {data.current_condition[0].temp_C}°C")
```

With the metadata feature, the package will always return usable data (falling back to mock data) rather than error strings, making it more resilient during educational use, such as when multiple students are accessing the service simultaneously.

## Architecture Details

The package has a modular architecture:

```
src/fetch_my_weather/
├── __init__.py      # Exports public API
├── core.py          # Core implementation
└── models.py        # Pydantic models for JSON data
```

Data flow:
1. User calls `get_weather()` with parameters
2. If mock mode is enabled and not explicitly disabled for this request, mock data is returned
3. Parameters are validated
4. URL is constructed using `_build_url()`
5. Cache is checked using `_get_from_cache()`
6. If data is not in cache, HTTP request is made
7. Response is processed (with JSON data converted to Pydantic models)
8. Response is stored in cache using `_add_to_cache()`
9. Data is returned to the user

The caching system uses a simple in-memory dictionary with URL keys and (timestamp, data) values.

### Pydantic Models

The package uses Pydantic models to provide structured, type-safe access to JSON weather data, including:

- `WeatherResponse`: The top-level response model
- `CurrentCondition`: Current weather information
- `DailyForecast`: Daily weather forecast
- `HourlyForecast`: Hourly weather forecast
- `NearestArea`: Location information
- `Request`: Information about the API request (query string and location type)
- `Astronomy`: Sun and moon information
- `ResponseMetadata`: Metadata about the response (real/cached/mock, error info)
- `ResponseWrapper`: Container for both data and metadata

## Common Use Cases

1. **Simple Weather Check**: Get current location weather
2. **Multi-City Weather Comparison**: Get weather for multiple cities and compare
3. **Weather Images**: Generate weather images for display
4. **Moon Phase Information**: Get current or future moon phases
5. **Weather in Different Languages**: Get weather information in various languages
6. **Structured JSON Data**: Process weather data with type safety using Pydantic models
7. **Response Metadata**: Check data source and handle errors gracefully
8. **Development and Testing**: Use mock mode to develop without hitting API rate limits
9. **Educational Projects**: Using the package to teach API interactions and data processing

## Project Ideas

The package includes mini-projects of varying difficulty levels:

### Beginner Projects
- Personal Weather Dashboard
- Multi-City Weather Checker
- Weather Image Saver

### Intermediate Projects
- Weather Mood Recommender
- Weekly Weather Forecast Tracker
- Weather-based Wallpaper Changer

### Advanced Projects
- Weather Notification System
- Weather Data Logger and Analyzer
- Weather-Based Home Automation
- Weather-based Game World Generator

## Best Practices

1. **Check for Errors**: Always check if the result starts with "Error:" before using it
2. **Use Caching**: Leverage the built-in caching to avoid unnecessary requests
3. **Respect the Service**: Don't make too many requests to wttr.in
4. **Use Mock Mode for Development**: Enable mock mode during development and testing
5. **Use JSON Format with Models**: Prefer the JSON format with Pydantic models for type safety
6. **Handle PNG Returns**: Remember that PNG requests return bytes, not strings
7. **Use Compact Views**: Use view_options="0" or "1" for smaller, more focused weather data

## Key Limitations

1. **In-memory Cache**: Cache is not persistent across program restarts
2. **Network Dependency**: Requires internet connection to fetch real weather data (though mock mode can help)
3. **Service Limitations**: Subject to wttr.in service availability and rate limits
4. **Fixed Pydantic Model Structure**: The Pydantic models match wttr.in's output format, which may change

## Acknowledgments

This package is a wrapper around the [wttr.in](https://github.com/chubin/wttr.in) service created by [Igor Chubin](https://github.com/chubin).