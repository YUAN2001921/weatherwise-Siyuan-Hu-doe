---
title: "Before/After Example: Weather Data Retrieval Function"
author: "Michael Borck"
format: 
  pdf:
    toc: false
    colorlinks: true
  docx:
    toc: false
    highlight-style: github
  html:
    toc: true
    toc-expand: 2
    embed-resources: true
---


## Initial Prompt

```
Write a function to retrieve weather data from the wttr.in API for a given location.
```

## Initial AI-Generated Code (Before)

```python
def get_weather(location):
    import requests
    
    url = f"https://wttr.in/{location}?format=j1"
    response = requests.get(url)
    data = response.json()
    
    return data
```

## My Analysis of the Initial Code

The initial code has several issues:

1. No error handling - will crash if the API is down or returns an error
2. No input validation for the location parameter
3. Returns the raw API response which is complex and includes unnecessary data
4. No timeout parameter which could cause the application to hang
5. No docstring or comments explaining how to use the function

## My Follow-up Prompt

```
Thanks for the starting point. I'd like to improve this function with:

1. Error handling for API failures and bad responses
2. Location input validation
3. A timeout parameter to prevent hanging
4. A way to process the response to return only essential weather information
5. Proper documentation with parameters and return values explained

Also, can you make it handle the forecast_days parameter to allow getting forecast for multiple days?
```

## Improved AI-Generated Code (After)

```python
def get_weather_data(location, forecast_days=5, timeout=10):
    """
    Retrieve weather data for a specified location from wttr.in API.
    
    Args:
        location (str): City or location name
        forecast_days (int): Number of days to forecast (1-5)
        timeout (int): Timeout for API request in seconds
        
    Returns:
        dict: Weather data including current conditions and forecast,
             or None if an error occurred
    """
    import requests
    
    # Input validation
    if not location or not isinstance(location, str):
        print("Error: Location must be a non-empty string")
        return None
    
    if not 1 <= forecast_days <= 5:
        print("Error: forecast_days must be between 1 and 5")
        forecast_days = min(max(1, forecast_days), 5)  # Clamp to valid range
    
    # Clean up the location string
    location = location.strip().replace(" ", "+")
    
    try:
        # Make request to wttr.in API
        url = f"https://wttr.in/{location}?format=j1"
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse JSON response
        data = response.json()
        
        # Process and extract relevant information
        result = {
            "location": location.replace("+", " "),
            "current": {
                "temperature": {
                    "current": int(data["current_condition"][0]["temp_C"]),
                    "feels_like": int(data["current_condition"][0]["FeelsLikeC"])
                },
                "condition": data["current_condition"][0]["weatherDesc"][0]["value"],
                "humidity": int(data["current_condition"][0]["humidity"]),
                "wind": {
                    "speed": int(data["current_condition"][0]["windspeedKmph"]),
                    "direction": data["current_condition"][0]["winddir16Point"]
                },
                "precipitation": float(data["current_condition"][0]["precipMM"])
            },
            "forecast": []
        }
        
        # Add forecast data
        for i in range(min(forecast_days, len(data["weather"]))):
            day = data["weather"][i]
            forecast_day = {
                "date": day["date"],
                "max_temp": int(day["maxtempC"]),
                "min_temp": int(day["mintempC"]),
                "condition": day["hourly"][4]["weatherDesc"][0]["value"],  # Midday condition
                "precipitation": {
                    "chance": int(day["hourly"][4]["chanceofrain"]),
                    "amount": float(day["hourly"][4]["precipMM"])
                },
                "wind": {
                    "speed": int(day["hourly"][4]["windspeedKmph"]),
                    "direction": day["hourly"][4]["winddir16Point"]
                }
            }
            result["forecast"].append(forecast_day)
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving weather data: {e}")
        return None
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error processing weather data: {e}")
        return None
```

## Why My Prompting Strategy Was Effective

My follow-up prompt was effective because:

1. **I identified specific problems** in the initial code rather than just asking for "better code"
2. **I requested specific improvements** with clear objectives (error handling, validation, etc.)
3. **I suggested a new feature** (forecast_days parameter) that would make the function more useful
4. **I asked for proper documentation** which resulted in comprehensive docstrings

This approach guided the AI to generate a much more robust function that:

- Is properly documented
- Validates inputs
- Has comprehensive error handling
- Returns structured, processed data instead of raw API response
- Includes the new forecast functionality I requested

The key to effective prompting was being specific about what needed improvement and why, rather than making vague requests for "better" code.
