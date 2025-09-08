---
title: "WeatherWise: Intelligent Weather Analysis & Advisory System"
subtitle: "Harnessing Python and AI to Create Intuitive Weather Applications"
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

## Overview

In this assignment, you will develop a Python application called "Weather Advisor" that combines weather data access with conversational AI capabilities. This project brings together several key concepts you've learnt throughout the course, including modular programming, data visualisation, user interfaces, and—crucially—intentional prompting techniques when working with AI.

Your solution should demonstrate both your technical Python skills and your ability to effectively direct AI tools as part of your development workflow. This project will challenge you to apply your skills creatively, and you'll end up with a practical application you can be proud of.

> **Note on Assignment Length**: This specification is comprehensive to accommodate various workflows and AI tools. Don't be intimidated by its length—it's designed to provide clarity and flexibility for all students, regardless of which AI tools you prefer to use. The detailed requirements help you understand how to demonstrate your process of working with AI, which is a key component of this assignment.

> **Embracing the Challenge**: This project may initially seem ambitious, but remember that you've already learnt all the fundamental techniques needed to succeed. The weekly activities and mini-projects have prepared you with the building blocks—from working with APIs to creating chatbots to visualising data. This assignment brings these pieces together into something greater than the sum of its parts. The satisfaction of overcoming this challenge and creating something genuinely useful will be well worth the effort!

## Learning Objectives

By completing this assignment, you will:

- Implement a complete, functional Python application with modular design
- Apply data visualisation techniques to weather information
- Create intuitive user interfaces using console-based menus or widgets
- Demonstrate your ability to use intentional prompting with AI tools
- Practice version control using GitHub
- Document your development process and AI interactions
- Apply critical thinking when evaluating and improving AI-generated code

## Project Requirements

### Core Features (Required)

Your Weather Advisor application must include:

1. **Weather Data Component**
   - Retrieve weather data for user-specified locations
   - Display current conditions and forecasts
   - Include at least 2 different data visualisations (e.g., temperature trends, precipitation chances)

2. **Natural Language Interface**
   - Allow users to ask weather-related questions in plain English
   - Parse these questions to determine the weather information needed
   - Provide natural-language responses that answer the user's question

3. **User Experience**
   - Implement an intuitive menu system using pyinputplus
   - Include clear instructions and error handling
   - Provide a clean, organised display of information

### Technical Requirements

Your implementation must demonstrate:

1. **Modular Design**
   - Break your program into logical functions and modules
   - Use appropriate naming conventions
   - Include docstrings and comments

2. **Core Functions**
   - Your application should implement these functions (provided as stubs in the starter notebook):
   
   ```python
   def get_weather_data(location, forecast_days=5):
       """
       Retrieve weather data for a specified location.
       
       Args:
           location (str): City or location name
           forecast_days (int): Number of days to forecast (1-5)
           
       Returns:
           dict: Weather data including current conditions and forecast
       """
       pass
       
   def parse_weather_question(question):
       """
       Parse a natural language weather question.
       
       Args:
           question (str): User's weather-related question
           
       Returns:
           dict: Extracted information including location, time period, and weather attribute
       """
       pass
       
   def generate_weather_response(parsed_question, weather_data):
       """
       Generate a natural language response to a weather question.
       
       Args:
           parsed_question (dict): Parsed question data
           weather_data (dict): Weather data
           
       Returns:
           str: Natural language response
       """
       pass
       
   def create_temperature_visualisation(weather_data, output_type='display'):
       """
       Create visualisation of temperature data.
       
       Args:
           weather_data (dict): The processed weather data
           output_type (str): Either 'display' to show in notebook or 'figure' to return the figure
           
       Returns:
           If output_type is 'figure', returns the matplotlib figure object
           Otherwise, displays the visualisation in the notebook
       """
       pass
       
   def create_precipitation_visualisation(weather_data, output_type='display'):
       """
       Create visualisation of precipitation data.
       
       Args:
           weather_data (dict): The processed weather data
           output_type (str): Either 'display' to show in notebook or 'figure' to return the figure
           
       Returns:
           If output_type is 'figure', returns the matplotlib figure object
           Otherwise, displays the visualisation in the notebook
       """
       pass
   ```

3. **User Interface Flexibility**
   - You may implement your user interface using either:
     - Console-based menus with pyinputplus
     - Interactive widgets using ipywidgets
     - A hybrid approach combining both
   - Your UI should call the core functions rather than duplicating their functionality
   - Choose the approach that best showcases your application

4. **Data Handling**
   - Retrieve and process weather data
   - Store relevant information for use in the application
   - Handle edge cases (e.g., invalid locations, missing data)

5. **Visualisation**
   - Create at least 2 different types of visualisations
   - Ensure visualisations are properly labelled and clear
   - Make visualisations relevant to the user's query when possible

### Weather Data Source Options

You have three options for implementing the weather data component. All options can earn full marks for the core requirements, but more advanced implementations may qualify for bonus points.

1. **Foundation Option**: Use the provided fetch-my-weather package
   - A simplified interface to wttr.in weather services that handles API complexities
   - Provides consistent data structures and error handling
   - Allows you to focus on application logic and other components
   - *Perfect for getting started and meeting core requirements*

2. **Standard Option**: Use the wttr.in service API directly
   - Example: `https://wttr.in/London?format=j1` returns JSON data
   - Documentation: https://github.com/chubin/wttr.in
   - Requires implementing your own request handling and data processing
   - *Demonstrates API interaction skills and offers more customisation*

3. **Extension Option**: Implement your own solution with OpenWeatherMap or similar
   - Requires obtaining and securely handling an API key
   - Offers access to more detailed weather data
   - Provides experience with commercial API integration
   - *Eligible for bonus points if well-implemented with proper error handling*

**Note on Bonus Points**: Students who implement their own weather data retrieval (options 2 or 3) with additional features beyond basic requirements may receive bonus consideration (up to 5% extra). Examples of bonus-worthy features include:
- Robust caching to minimise API calls
- Comprehensive error handling with informative user feedback
- Support for multiple weather data sources with a consistent interface
- Advanced data processing that enhances the user experience

Remember that your choice should align with your comfort level and project goals. All options can achieve full marks for the core requirements when implemented correctly, so choose the approach that you feel most confident with!

## Development Process Requirements

### GitHub Integration

- Create a GitHub repository for your project
- Make regular, meaningful commits throughout development
- Your final submission should have at least 15 commits that show progression
- Include a clear README.md file with instructions for running your application

### Intentional Prompting Documentation

A critical component of this assignment is documenting your use of AI tools:

1. **Implementation Options Exploration**
   - Include an initial AI conversation where you explore and compare the three implementation options
   - This conversation should demonstrate your analytical thinking about the project requirements and should go deeper than simply asking which option is best
   - Show how you evaluated the technical considerations, design implications, and alignment with your skills for each option
   - Your conversation should reflect a thoughtful decision-making process that led to your final choice

2. **Required Prompting Techniques**
   You must demonstrate at least 5 of the following intentional prompting techniques:
   
   - **Restate the problem** in your own words to the AI
   - **Identify input/output requirements** through targeted prompts
   - **Request pseudocode** before asking for implementation
   - **Challenge edge cases** in AI-generated code
   - **Request modular design** improvements
   - **Ask for code explanations** to ensure understanding
   - **Request iterative improvements** to initial solutions
   - **Query design trade-offs** for different implementation options

3. **Before/After Examples**
   - Include at least 3 clear examples showing:
     - Initial AI-generated code from your prompt
     - Your specific follow-up prompts that led to improvements
     - The improved code that resulted from your intentional prompting
     - Explanation of why your prompting strategy was effective

### AI Conversation Documentation Requirements

You must document your AI interactions using the following format:

1. **File Format**
   - Save conversations as plain text files (`.txt`)
   - Name files sequentially: `conversation1.txt`, `conversation2.txt`, etc.
   - Submit at least 5 significant conversations that demonstrate different aspects of your development process

2. **Content Requirements**
   - Include a brief header for each conversation (e.g., "Conversation about weather data retrieval")
   - Clearly indicate who is speaking (e.g., "Me:", "AI:")
   - Include the full conversation with both your prompts and the AI's responses
   - Add brief comments or notes where relevant to highlight important points

3. **Important Guidelines**
   - Copy conversations directly from the chat interface
   - Include complete interactions, not just selected snippets
   - Do not submit screenshots or images of conversations
   - If a conversation is very long, you may focus on the most relevant sections, but indicate where content has been omitted
   - Include at least one conversation showing how you handled an incorrect or problematic AI response

### Documentation for Alternative AI Tools

If you use tools like GitHub Copilot, Cursor, Whisper, Claude-Code, or other coding assistants that don't provide traditional conversation logs:

1. **Screen Recording Option**
   - You may submit brief screen recordings (2-5 minutes each) showing your interaction with these tools
   - The recordings should clearly show both your inputs and the AI's suggestions
   - Include a brief narration or text overlay explaining what you're doing and how you're directing the AI

2. **Manual Documentation Option**
   - Create a log that approximates the conversation by documenting:
     - What you asked or prompted (partial code, comments, etc.)
     - What the AI suggested (completions, alternatives, etc.)
     - How you evaluated and refined those suggestions
   - Structure this as a series of exchanges, similar to a chat conversation
   - Include screenshots of the interaction where helpful, but these should supplement text, not replace it

3. **Tool-Specific Requirements**
   - **GitHub Copilot**: Document your use of comments to guide suggestions and how you refined them
   - **Cursor**: You may use the chat sidebar for more conversational interactions and submit those logs
   - **Whisper or voice-based tools**: Transcribe key voice commands and responses
   - **Claude-Code and similar tools**: Capture the prompts and responses

4. **Multilingual Considerations**
   - If English is not your first language, you may conduct initial AI conversations in your preferred language
   - Submit both the original conversation in your language and an English translation
   - All code, comments, documentation, and final submission materials must be in English
   - Consider using the six-step methodology in your native language for planning, then translate when implementing

Regardless of the AI tool you use, you must still demonstrate intentional prompting by showing how you guided the tool to help you solve problems rather than accepting its first suggestions. All students must include the required "before/after" examples regardless of which AI tools they use.

Conversations must be submitted as searchable text (or as described above for alternative tools), not images. Submissions that use screenshots or non-text formats for conversations will receive significant reduced marks for that component.

## Project Implementation Options

You may structure your project in one of the following ways:

### Option 1: Advanced Weather Dashboard
- Create a comprehensive weather dashboard with multiple visualisation types
- Allow filtering by time periods and weather attributes
- Include a chat interface for natural language queries about displayed data

### Option 2: Weather-Aware Chatbot
- Focus on the conversational interface as the primary interaction method
- Retrieve and display weather information based on natural language analysis
- Use visualisations to supplement text responses when appropriate

### Option 3: Hybrid Approach (Recommended)
- Implement both dashboard and conversational features
- Allow users to switch between interaction modes
- Ensure visualisation and chat components are well-integrated

## Notebook Organisation Requirements

Your Google Colab notebook should demonstrate clear organisation through a cell-based modular approach:

### Cell-Based Structure
- Organise your notebook into clearly labelled sections using markdown cells
- Group related functionality together in dedicated sections
- Include a table of contents at the top of the notebook

### Required Sections
Your notebook should include at least these labelled sections:
1. **Setup and Configuration** - Imports and initial setup
2. **Weather Data Functions** - Functions for retrieving and processing weather data
3. **Visualisation Functions** - Code for creating data visualisations
4. **Natural Language Processing** - Functions for handling user questions
5. **User Interface** - Menu system and display functions
6. **Main Application Logic** - Core functionality that ties everything together
7. **Testing and Examples** - Demonstrations of key features

### Modular Design
- Define general-purpose functions that can be reused
- Minimise code duplication
- Keep functions focused on single responsibilities
- Include appropriate comments and docstrings

The organisation of your notebook will be part of your assessment, with particular attention to:
- Logical grouping of related functionality
- Clear section headings and descriptions
- Appropriate separation of concerns
- Documentation quality


### Weather Data API Testing

When testing your application, we will use mock versions of the weather API services. To ensure compatibility:

1. Your `get_weather_data()` function should follow the provided signature
2. If using OpenWeatherMap, access the API key via: `os.environ.get("OPENWEATHER_API_KEY")`
3. Your code should handle cases where the API returns incomplete data
4. For testing, we will simulate various weather conditions and error cases
5. Ensure your code can handle both successful and failed API calls gracefully
6. Use the `requests` library for API calls, and ensure you handle exceptions properly


## Submission Requirements

Your submission should include:

1. **GitHub Repository**
   - All project files must be uploaded to your GitHub repository
   - Invite your instructor to your repository (GitHub username: michael-borck)
   - Your repository should include:
     - Google Colab notebook (.ipynb file)
     - README.md with project overview and setup instructions
     - AI conversation text files (conversation1.txt, conversation2.txt, etc.)
     - Any additional supporting documentation you wish to include
   - Ensure the repository is properly organised and complete

2. **LMS Submission**
   - Download a ZIP file of your entire GitHub repository
   - Submit this ZIP file to the course LMS by the due date
   - Ensure all files are included and the ZIP can be extracted properly

3. **Required Files**
   - Google Colab notebook with your complete, working application
   - Documentation files as specified above
   - Text files of AI conversations
   - A brief reflection document (300-500 words) on your development process

4. **Code Quality**
   - Well-commented code with docstrings
   - Consistent naming conventions
   - Proper error handling

5. **Testing Resource**
   - A preliminary version of your notebook can be submitted to the course testing service
   - The testing service will check basic functionality and notebook structure
   - Submission to the testing service is optional but recommended to identify issues early

**Note**: Submitting both to GitHub and the LMS ensures we have access to your work even if there are technical issues with either platform.

## Assessment Criteria

| Criterion | Weighting | Description |
|-----------|-----------|-------------|
| **Functionality** | 15% | Application works as expected; features are complete (demonstrate development process in AI conversations) |
| **Code Quality** | 10% | Well-structured, readable code with proper documentation (show code refinement through AI conversations) |
| **Notebook Organisation** | 10% | Logical cell-based structure; clear section organisation (reflect planning discussions in AI conversations) |
| **User Experience** | 10% | Interface is intuitive and information is clearly presented (show UI design iterations in AI conversations) |
| **Intentional Prompting** | 30% | Documentation of AI interactions; evidence of strategic prompting techniques that improved your implementation |
| **AI Conversation Quality** | 15% | Comprehensive conversation documentation showing problem-solving journey and thoughtful interactions |
| **Technical Implementation** | 10% | Appropriate use of libraries and efficient implementation (demonstrate technical decision-making in AI conversations) |

Remember, this project is your opportunity to showcase everything you've learnt in this course. We're excited to see your creative solutions!

## Extensions (Optional for Higher Grades)

By completing the core requirements, you'll demonstrate solid mastery of the course material. If you're looking for additional challenges, consider these extensions:

- Historical weather data analysis with more advanced visualisations
- Location-based recommendations (e.g., "Is tomorrow good for hiking?")
- Integration with calendar or event planning features
- Advanced error handling and input validation
- Custom themes or visualisation styles

## Support Resources

- Weather API documentation (links to be provided)
- hands-on-ai package documentation
- Sample code for basic weather data retrieval
- Office hours for technical questions

## Due Date

Submit your completed assignment by 23:59 on Friday 23rd May 2025.

## Notes on Academic Integrity

While you are encouraged to use AI tools as part of your development process, your submission must represent your own understanding and effort. You must be able to explain all aspects of your code, including any AI-generated portions. The intentional prompting documentation is a critical component that demonstrates your engagement with the material.

This assignment assesses not just the final product, but your process and interaction with AI tools. The detailed documentation requirements ensure that your unique contribution is visible regardless of which AI tools you use. Remember that effectively guiding AI tools is a skill itself that requires critical thinking and domain knowledge.

Good luck, and enjoy building your Weather Advisor application!
