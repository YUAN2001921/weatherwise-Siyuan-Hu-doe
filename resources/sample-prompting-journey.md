---
title: "Sample Intentional Prompting Journey"
subtitle: "Weather Dashboard Project"
format:
  pdf:
    toc: false
    number-sections: false
    colorlinks: true
  docx:
    toc: false
    number-sections: false
    highlight-style: github
  html:
    toc: true
    toc-expand: 2
    embed-resources: true
---

This document demonstrates how a student might approach the Weather Dashboard project using intentional prompting techniques. It follows the 6-step programming methodology and showcases a variety of prompt types.

> **Important Note to Students**: This document demonstrates the *approach* to intentional prompting, not specific prompts to copy. Your project will have unique requirements and challenges that will require your own thoughtful prompts. Simply copying these example prompts will not result in the best solution for your specific implementation, and will not demonstrate your personal understanding and engagement with the AI. The assessment focuses on your ability to craft effective prompts specific to your project's needs - not on reproducing this example. Use this as inspiration for your own authentic conversations with AI, where you identify problems, ask clarifying questions, challenge initial solutions, and guide the AI toward better outcomes for your particular implementation.

---

## Step 1: Restate the Problem (Foundation Prompts)

### 🗣️ Initial Prompt
```
I need to create a weather dashboard in Python that uses a provided module to access weather data and displays it in a user-friendly interface. Can you help me restate this problem more clearly to ensure I understand all requirements?
```

### 🤖 AI Response (partial)
> This problem involves creating a Python application that:
> 1. Utilises an existing weather data module to retrieve weather information
> 2. Processes that data appropriately
> 3. Presents the information in a clear, user-friendly interface
> 4. Allows users to interact with the dashboard to view different weather metrics
>
> The key components you'll need to develop are:
> - Functions to retrieve and process data from the provided module
> - A display system (likely console-based) to show weather information
> - User interaction components to allow navigation and selection
> - Visualisation elements for presenting weather data graphically

### 🗣️ Follow-up Prompt (Clarification)
```
That's helpful. Could you help me understand what types of weather data are typically available through weather APIs, and what visualisations might be most useful for users? This will help me plan my dashboard layout.
```

### 🤖 AI Response (partial)
> Typical weather data available through APIs includes:
> - Current temperature (actual, feels like)
> - Temperature range (high/low)
> - Humidity percentage
> - Wind speed and direction
> - Precipitation probability and amount
> - Atmospheric pressure
> - UV index
> - Visibility distance
> - Weather conditions (sunny, cloudy, rainy, etc.)
> - Forecasts (hourly, daily)
>
> For visualisations, these are particularly useful:
> - Line graphs for temperature trends over time
> - Bar charts for comparing high/low temperatures
> - Gauge charts for humidity, UV index
> - Wind direction indicators (compass or arrow)
> - Icons for weather conditions
> - Radar maps for precipitation (if mapping capabilities are available)

## Step 2: Identify Input/Output Requirements

### 🗣️ Requirement Gathering Prompt
```
Based on our discussion, let me identify the specific inputs and outputs for my weather dashboard:

Inputs:
- Location information (city name or coordinates)
- Type of weather data requested (current, forecast)
- Time period for forecast data

Outputs:
- Current weather conditions display
- Visual representations of temperature, humidity, etc.
- Forecast for upcoming days
- User navigation menu

Does this cover the essential inputs and outputs? Is there anything important I'm missing?
```

### 🤖 AI Response
> Your input/output identification is strong! A few additional considerations:
>
> Additional inputs to consider:
> - User preferences for temperature units (Celsius/Fahrenheit)
> - Display preferences (text-only vs. graphical visualisation)
> - Refresh rate for data updates
>
> Additional outputs to consider:
> - Error messages for handling unavailable data or connection issues
> - Status indicators for data freshness
> - Historical data comparisons (if available)
> - Weather alerts or warnings
>
> Also, since you mentioned a provided module, you'll need to understand:
> - What function calls are available in the module
> - What data format it returns
> - Any authentication or rate limiting requirements

## Step 3: Work Problem by Hand

### 🗣️ Manual Solution Prompt
```
Before we start coding, I want to sketch out the user interaction flow for the dashboard:

1. User starts program
2. Program displays welcome and prompts for location
3. User enters location
4. Program fetches current weather data
5. Program displays main dashboard with:
   - Current temperature and conditions
   - Today's high/low
   - Humidity and wind
   - Simple 3-day forecast
6. Program shows menu options:
   - View detailed forecast
   - View temperature graph
   - Change location
   - Exit
7. User selects option
8. Program displays selected information
9. Return to menu (step 6)

Does this flow make sense? Would you structure the user experience differently?
```

### 🤖 AI Response
> Your flow makes excellent sense and covers the core user journey! Here are some refinements to consider:
>
> 1. Add an initial option to select temperature units (C/F) or use a default
> 2. Consider adding a "refresh data" option in your menu
> 3. You might want to include error handling flows:
>    - What happens if location isn't found?
>    - What if the weather service is unavailable?
>    - How to handle invalid user input?
>
> For the dashboard display, consider:
> - Using ASCII art for simple visualisation in console
> - Having a "summary view" and "detailed view"
> - Including timestamps to show when data was last updated
>
> The overall structure is logical and user-friendly!

## Step 4: Request Pseudocode

### 🗣️ Pseudocode Prompt

```text
Now that we have a clear understanding of the problem and user flow, could you help me write pseudocode for this weather dashboard? I want to focus on creating a modular design with separate functions for data retrieval, display, and user interaction.
```

### 🤖 AI Response
```python
# Main program structure
function main():
    display_welcome()
    location = get_location_from_user()
    units = get_preferred_units()
    
    while True:
        weather_data = fetch_weather_data(location, units)
        display_dashboard(weather_data)
        choice = display_menu()
        
        if choice == "forecast":
            display_detailed_forecast(weather_data)
        else if choice == "graph":
            display_temperature_graph(weather_data)
        else if choice == "location":
            location = get_location_from_user()
        else if choice == "refresh":
            continue  # Will loop back and refresh data
        else if choice == "exit":
            break

# Data functions
function fetch_weather_data(location, units):
    # Use provided weather module to get data
    current_weather = weather_module.get_current(location, units)
    forecast = weather_module.get_forecast(location, units)
    return {current: current_weather, forecast: forecast}

# Display functions
function display_dashboard(weather_data):
    clear_screen()
    display_current_conditions(weather_data["current"])
    display_simple_forecast(weather_data["forecast"])

function display_detailed_forecast(weather_data):
    clear_screen()
    for day in weather_data["forecast"]:
        display_day_forecast(day)
    wait_for_user()

function display_temperature_graph(weather_data):
    clear_screen()
    temperatures = extract_temperatures(weather_data["forecast"])
    dates = extract_dates(weather_data["forecast"])
    create_line_graph(dates, temperatures)
    wait_for_user()

# User interaction functions
function display_menu():
    print menu options
    return get_valid_choice()

function get_location_from_user():
    prompt for location
    validate location
    return location
```

### 🗣️ Follow-up Prompt (Challenge)
```
The pseudocode looks good overall. I noticed that error handling isn't explicitly included. Could you expand the pseudocode to include error handling for:

1. Invalid location input
2. Weather service connection failures
3. Handling missing data in the weather response

Also, how would you structure the functions to make testing easier?
```

## Step 5: Convert to Python Code

### 🗣️ Initial Implementation Prompt
```
Let's start converting this to Python code. First, could you implement the main() function and the user interaction functions? Assume the weather module is imported as 'weather_api' and has functions:

- weather_api.get_current(location, units) - returns current weather
- weather_api.get_forecast(location, units, days=3) - returns forecast

I'll be using pyinputplus for user input validation and matplotlib for any visualisations.
```

### 🤖 AI Response (partial code)
```python
import pyinputplus as pyip
import matplotlib.pyplot as plt
import weather_api
import os
import time

def main():
    """Main function to run the weather dashboard."""
    display_welcome()
    location = get_location_from_user()
    units = get_preferred_units()
    
    while True:
        try:
            weather_data = fetch_weather_data(location, units)
            display_dashboard(weather_data)
            choice = display_menu()
            
            if choice == "forecast":
                display_detailed_forecast(weather_data)
            elif choice == "graph":
                display_temperature_graph(weather_data)
            elif choice == "location":
                location = get_location_from_user()
            elif choice == "refresh":
                continue  # Will loop back and refresh data
            elif choice == "exit":
                print("Thank you for using the Weather Dashboard!")
                break
        except ConnectionError:
            print("Unable to connect to weather service. Please check your internet connection.")
            time.sleep(2)
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(2)

def display_welcome():
    """Display welcome message."""
    clear_screen()
    print("=" * 60)
    print("               WEATHER DASHBOARD")
    print("=" * 60)
    print("\nWelcome to the Weather Dashboard!")
    print("This application provides current weather and forecasts.")
    print("\nPress Enter to continue...")
    input()

def get_location_from_user():
    """Get and validate location from user."""
    while True:
        clear_screen()
        print("Location Entry")
        print("-" * 60)
        try:
            location = pyip.inputStr("Enter city name (e.g., 'New York'): ")
            # Here we would typically validate the location by making a test API call
            # For now, we'll just check if it's not empty
            if location.strip():
                return location
            print("Please enter a valid location.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            exit()

def get_preferred_units():
    """Get user's preferred temperature units."""
    clear_screen()
    print("Temperature Units")
    print("-" * 60)
    units_choice = pyip.inputMenu(["Celsius", "Fahrenheit"], 
                                 numbered=True,
                                 prompt="Select your preferred temperature units:\n")
    return "metric" if units_choice == "Celsius" else "imperial"

def display_menu():
    """Display and handle main menu."""
    options = ["View detailed forecast", 
               "View temperature graph",
               "Change location",
               "Refresh data",
               "Exit"]
    
    choice = pyip.inputMenu(options, 
                          numbered=True,
                          prompt="\nWhat would you like to do?\n")
    
    # Convert menu choice to internal command
    if choice == "View detailed forecast":
        return "forecast"
    elif choice == "View temperature graph":
        return "graph"
    elif choice == "Change location":
        return "location"
    elif choice == "Refresh data":
        return "refresh"
    else:
        return "exit"

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def wait_for_user():
    """Wait for user to press Enter to continue."""
    print("\nPress Enter to return to the main menu...")
    input()
```

### 🗣️ Refinement Prompt

```
This looks good as a starting point. Now I need to implement the data functions and display functions. Can you help me implement:

1. fetch_weather_data()
2. display_dashboard()
3. display_detailed_forecast()
4. display_temperature_graph()

For the graph, I want to show a line graph of temperatures (high/low) for the forecast days. Please add appropriate error handling for API connection issues.
```

### 🤖 AI Response (partial code)
```python
def fetch_weather_data(location, units):
    """
    Fetch weather data from API.
    
    Args:
        location: City name or location
        units: 'metric' or 'imperial'
        
    Returns:
        Dictionary containing current weather and forecast
        
    Raises:
        ConnectionError: If unable to connect to weather service
        ValueError: If location is invalid
    """
    try:
        # Attempt to get current weather
        current_weather = weather_api.get_current(location, units)
        
        # Attempt to get forecast
        forecast = weather_api.get_forecast(location, units, days=5)
        
        # Basic validation of returned data
        if not current_weather or 'temperature' not in current_weather:
            raise ValueError(f"Invalid data received for location: {location}")
        
        return {"current": current_weather, "forecast": forecast}
    
    except ConnectionError:
        raise ConnectionError("Failed to connect to weather service")
    except Exception as e:
        # Convert any other exceptions from the API to something we can handle
        raise Exception(f"Error fetching weather data: {str(e)}")

def display_dashboard(weather_data):
    """
    Display the main weather dashboard.
    
    Args:
        weather_data: Dictionary containing current weather and forecast
    """
    clear_screen()
    current = weather_data["current"]
    forecast = weather_data["forecast"]
    
    # Display header and current conditions
    print("=" * 60)
    print(f"  WEATHER DASHBOARD - {current['location'].upper()}")
    print("=" * 60)
    print(f"\nCurrent Conditions as of {current['timestamp']}")
    print(f"Temperature: {current['temperature']}°{'C' if current['units'] == 'metric' else 'F'}")
    print(f"Feels Like: {current['feels_like']}°{'C' if current['units'] == 'metric' else 'F'}")
    print(f"Condition: {current['condition']}")
    print(f"Humidity: {current['humidity']}%")
    print(f"Wind: {current['wind_speed']} {'km/h' if current['units'] == 'metric' else 'mph'} from {current['wind_direction']}")
    
    # Display simplified forecast
    print("\n3-DAY FORECAST")
    print("-" * 60)
    
    for i, day in enumerate(forecast[:3]):
        day_label = "Today" if i == 0 else "Tomorrow" if i == 1 else day['date']
        print(f"{day_label}: High {day['high']}° | Low {day['low']}° | {day['condition']}")
```

### 🗣️ Integration Prompt
```
I need to complete the remaining display functions and integrate everything. Could you provide the code for:

1. display_detailed_forecast()
2. display_temperature_graph()

Also, I'm not seeing where the error handling for invalid locations is implemented. Could you improve the get_location_from_user() function to attempt validation through the weather API before accepting the input?
```

## Step 6: Test with Data

### 🗣️ Testing Prompt
```
Now I want to test this code with some sample data. Can you provide a sample data structure that would be returned by:

1. weather_api.get_current()
2. weather_api.get_forecast()

Also, what test cases should I consider to ensure my dashboard handles all possible scenarios correctly?
```

### 🤖 AI Response (partial)
> Here are sample data structures that might be returned by the weather API:
>
> ```python
> # Sample current weather data
> sample_current = {
>     'location': 'New York',
>     'timestamp': '2023-09-15 14:30',
>     'temperature': 22,
>     'feels_like': 24,
>     'condition': 'Partly Cloudy',
>     'humidity': 65,
>     'wind_speed': 8,
>     'wind_direction': 'NE',
>     'pressure': 1012,
>     'visibility': 10,
>     'uv_index': 5,
>     'units': 'metric'
> }
>
> # Sample forecast data (list of days)
> sample_forecast = [
>     {
>         'date': '2023-09-15',
>         'day_of_week': 'Friday',
>         'high': 24,
>         'low': 18,
>         'condition': 'Partly Cloudy',
>         'precipitation_chance': 20,
>         'humidity': 65,
>         'wind_speed': 8,
>         'wind_direction': 'NE',
>         'units': 'metric'
>     },
>     # Additional days would follow...
> ]
> ```
>
> Test cases to consider:
>
> 1. **Happy path**:
>    - Valid location with complete weather data
>    - Multiple days of forecast
>
> 2. **Data issues**:
>    - Missing fields in current weather
>    - Partial forecast data available
>    - Empty forecast list
>
> 3. **Connection issues**:
>    - API timeout
>    - Service unavailable
>
> 4. **User input scenarios**:
>    - Invalid location name
>    - Empty input
>    - Special characters in location name
>
> 5. **Display edge cases**:
>    - Very long location names
>    - Extreme temperature values
>    - Very long condition descriptions

### 🗣️ Edge Case Prompt
```
I'd like to specifically focus on handling the case where the weather service returns incomplete data. Could you show me how to modify the display_dashboard() function to gracefully handle missing fields in the weather data?

Also, the current implementation assumes a specific format for the weather data. How can I make it more robust to handle variations in the API response?
```

---

## Final Iteration and Before/After Example

### Before: Initial Data Retrieval Function
```python
def fetch_weather_data(location, units):
    """Fetch weather data from API."""
    current_weather = weather_api.get_current(location, units)
    forecast = weather_api.get_forecast(location, units, days=5)
    return {"current": current_weather, "forecast": forecast}
```

### 🗣️ Refinement Process Prompt
```
The fetch_weather_data() function needs improvement. It doesn't handle errors or validate the data. How can I make it more robust to:

1. Handle connection errors
2. Validate returned data structure
3. Handle missing or incomplete data
4. Provide meaningful error messages to the user
```

### After: Improved Data Retrieval Function
```python
def fetch_weather_data(location, units):
    """
    Fetch weather data from API with comprehensive error handling.
    
    Args:
        location: City name or location
        units: 'metric' or 'imperial'
        
    Returns:
        Dictionary containing current weather and forecast
        
    Raises:
        ConnectionError: If unable to connect to weather service
        ValueError: If location is invalid or data is incomplete
    """
    try:
        # Attempt to get current weather
        current_weather = weather_api.get_current(location, units)
        
        # Validate current weather data
        required_current_fields = ['temperature', 'condition', 'humidity', 'wind_speed']
        missing_fields = [field for field in required_current_fields if field not in current_weather]
        
        if missing_fields:
            # Handle missing fields by providing defaults
            for field in missing_fields:
                current_weather[field] = "N/A"
            
            # Log the issue but don't fail completely
            print(f"Warning: Some current weather data is unavailable: {', '.join(missing_fields)}")
        
        # Attempt to get forecast with retry mechanism
        max_retries = 3
        retry_count = 0
        forecast = None
        
        while retry_count < max_retries and forecast is None:
            try:
                forecast = weather_api.get_forecast(location, units, days=5)
                break
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Warning: Could not retrieve forecast data. Using current conditions only.")
                    forecast = []  # Empty forecast as fallback
                else:
                    time.sleep(1)  # Wait before retrying
        
        # Ensure forecast is a list (even if empty)
        if not isinstance(forecast, list):
            forecast = []
            
        return {
            "current": current_weather, 
            "forecast": forecast,
            "location": location,
            "units": units,
            "timestamp": time.strftime("%Y-%m-%d %H:%M")
        }
    
    except ConnectionError:
        raise ConnectionError("Failed to connect to weather service. Please check your internet connection.")
    except ValueError as e:
        raise ValueError(f"Invalid location: {location}. Please check spelling and try again.")
    except Exception as e:
        # Convert any other exceptions from the API to something we can handle
        raise Exception(f"Error fetching weather data: {str(e)}")
```

### Explanation
The improved function demonstrates:

1. **Comprehensive error handling** with specific error types
2. **Data validation** to check for required fields
3. **Graceful degradation** by providing defaults for missing data
4. **Retry mechanism** for transient failures
5. **Defensive programming** by ensuring forecast is always a list
6. **Informative error messages** for better user experience
7. **Additional metadata** (timestamp, location) for context

This improvement resulted from intentional prompting that specifically challenged the AI to address potential failure modes and edge cases, rather than accepting the initial implementation.
