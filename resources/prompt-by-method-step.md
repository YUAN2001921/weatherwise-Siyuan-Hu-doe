---
title: "🧠 Effective AI Prompts for Programming Concepts"
subtitle: "Organised by the Six Core Programming Foundations"
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

This guide provides targeted prompts for each of the six programming foundations from the handout. Each section includes:

- 🎯 Concept Exploration Prompts: For understanding the concept
- 🛠️ Implementation Prompts: For applying the concept
- 🔍 Debugging Prompts: For fixing issues
- 🚀 Challenge Prompts: For extending your learning

> **Important Note to Students**: This document demonstrates the *approach* to intentional prompting, not specific prompts to copy. Your project will have unique requirements and challenges that will require your own thoughtful prompts. Simply copying these example prompts will not result in the best solution for your specific implementation, and will not demonstrate your personal understanding and engagement with the AI. The assessment focuses on your ability to craft effective prompts specific to your project's needs - not on reproducing this example. Use this as inspiration for your own authentic conversations with AI, where you identify problems, ask clarifying questions, challenge initial solutions, and guide the AI toward better outcomes for your particular implementation.

---

## 1️⃣ INPUT: Getting Data Into Your Program

### 🎯 Concept Exploration Prompts
- "What are the different ways to get user input in Python?"
- "What's the difference between input() and command-line arguments?"
- "How should I validate user input to ensure it's the right type?"
- "When should I use try/except blocks with user input?"

### 🛠️ Implementation Prompts
- "Can you show me how to get a number from the user and validate it's a positive integer?"
- "How do I prompt for a password without showing the characters?"
- "What's the best way to get a yes/no response from a user?"
- "How can I create a menu system for user input choices?"

### 🔍 Debugging Prompts
- "My input() function is returning strings when I need numbers. How do I fix this?"
- "Users keep entering invalid data. How can I create a loop that repeats until they enter valid input?"
- "How can I handle the case where a user just presses Enter without typing anything?"

### 🚀 Challenge Prompts
- "Can you show me how to create a robust input system that handles different types (int, float, string) with appropriate validation?"
- "How would I implement tab-completion for user input?"
- "Can you demonstrate reading input from a file versus interactively from a user?"

---

## 2️⃣ OUTPUT: Displaying Results

### 🎯 Concept Exploration Prompts
- "What's the difference between print() and return in Python?"
- "How can I format numbers to show specific decimal places?"
- "What are f-strings and how do they improve output formatting?"
- "When should I write output to a file instead of the console?"

### 🛠️ Implementation Prompts
- "How can I create a nicely formatted table of data in the console?"
- "What's the best way to show progress during a long-running operation?"
- "Can you show me how to use color in terminal output?"
- "How do I align text to the right when printing?"

### 🔍 Debugging Prompts
- "Why does my print statement show 'None' after printing my value?"
- "My floating-point numbers are displaying too many decimal places. How do I fix this?"
- "I'm getting TypeError when using f-strings. What might I be doing wrong?"

### 🚀 Challenge Prompts
- "Can you show me how to create a custom progress bar for console applications?"
- "How would I implement a simple logging system instead of using print statements?"
- "Can you demonstrate creating a formatted report that could be output to console or saved to a file?"

---

## 3️⃣ STORE: Variable Management and Data Structures

### 🎯 Concept Exploration Prompts
- "When should I use a list versus a dictionary in Python?"
- "What's the difference between mutable and immutable data types?"
- "How do variable references work in Python compared to other languages?"
- "What is the scope of variables in functions versus global scope?"

### 🛠️ Implementation Prompts
- "How can I store a collection of unique items in order of insertion?"
- "What's the best way to represent a grid or 2D structure in Python?"
- "Can you show me how to create a nested dictionary for hierarchical data?"
- "How do I efficiently check if an item exists in a collection?"

### 🔍 Debugging Prompts
- "Why does my function modify the list even though I don't return it?"
- "I'm getting 'UnboundLocalError' when trying to modify a variable inside a function. Why?"
- "My dictionary keys aren't working as expected. What might be wrong?"
- "How can I find memory leaks or inefficient data storage in my code?"

### 🚀 Challenge Prompts
- "Can you show me how to implement a custom cache using dictionaries?"
- "How would I create a data structure to represent a family tree?"
- "Can you demonstrate when using classes would be better than dictionaries for storing complex data?"

---

## 4️⃣ CALCULATE: Operations and Expressions

### 🎯 Concept Exploration Prompts
- "What's the difference between `==` and `is` in Python?"
- "How do operator precedence rules work in complex expressions?"
- "When should I use bitwise operators versus logical operators?"
- "What are the performance implications of different mathematical operations?"

### 🛠️ Implementation Prompts
- "Can you show me how to calculate the average of a list of numbers?"
- "What's the most efficient way to check if a string contains any of several substrings?"
- "How do I implement a custom sorting key for complex objects?"
- "Can you demonstrate using lambda functions for calculations?"

### 🔍 Debugging Prompts
- "Why is my floating-point comparison not working as expected?"
- "My boolean logic seems backwards. How can I troubleshoot complex conditions?"
- "I'm getting integer division when I want float division. What's happening?"
- "How can I fix 'TypeError: unsupported operand type' errors?"

### 🚀 Challenge Prompts
- "Can you show me how to implement a simple calculator that evaluates expressions entered as strings?"
- "How would I create a function that finds all prime numbers below n using the Sieve of Eratosthenes?"
- "Can you demonstrate implementing a basic statistical analysis library (mean, median, mode, std dev)?"

---

## 5️⃣ DECISIONS: Flow Control and Conditionals

### 🎯 Concept Exploration Prompts
- "What's the difference between 'if-elif-else' and multiple 'if' statements?"
- "How does short-circuit evaluation work in logical expressions?"
- "When should I use a ternary operator versus a full if statement?"
- "What are good practices for nested conditionals to maintain readability?"

### 🛠️ Implementation Prompts
- "Can you show me how to implement a state machine using if statements?"
- "What's the best way to handle multiple conditions that might all be true?"
- "How do I check if a value is in a specific range efficiently?"
- "Can you demonstrate using dictionary mappings instead of long if-elif chains?"

### 🔍 Debugging Prompts
- "My if statement isn't being triggered even though the condition seems true. How can I debug this?"
- "I have deeply nested if statements that are hard to follow. How can I refactor this?"
- "Why does my code enter this branch when I expected it to skip it?"
- "How can I trace the execution path through complex conditional logic?"

### 🚀 Challenge Prompts
- "Can you show me how to implement a decision tree for a simple classification problem?"
- "How would I create a rules engine that evaluates complex business rules?"
- "Can you demonstrate building a simple expert system with conditional logic?"

---

## 6️⃣ REPEAT: Loops and Iteration

### 🎯 Concept Exploration Prompts
- "When should I use a for loop versus a while loop?"
- "What are list comprehensions and how do they compare to traditional loops?"
- "How do break, continue, and else clauses work in loops?"
- "What are the efficiency considerations for different iteration methods?"

### 🛠️ Implementation Prompts
- "Can you show me how to iterate through nested data structures?"
- "What's the best way to process items in batches instead of one by one?"
- "How do I iterate through multiple lists in parallel?"
- "Can you demonstrate using generators for memory-efficient iteration?"

### 🔍 Debugging Prompts
- "My loop seems to run forever. How can I debug infinite loops?"
- "I'm getting 'RuntimeError: dictionary changed sise during iteration'. What's happening?"
- "Why is my loop skipping certain elements?"
- "How can I optimise this loop that's running too slowly?"

### 🚀 Challenge Prompts
- "Can you show me how to implement a custom iterator class?"
- "How would I create a paginated data processor that works on very large datasets?"
- "Can you demonstrate using parallel processing to speed up a loop operation?"

---

## 🔄 Integration Prompts: Combining Concepts

These prompts help students see how the six foundations connect:

1. **Simple Integration**:
   - "Can you show me a program that gets user input, validates it, performs calculations, and displays formatted results?"
   - "How would I create a loop that asks for input until the user enters valid data?"

2. **Intermediate Integration**:
   - "Can you demonstrate a data processing pipeline that reads data, transforms it through multiple stages, and outputs analysis?"
   - "How would I implement a command-line interface with different operations on stored data?"

3. **Advanced Integration**:
   - "Can you show me how to build a text-based game that uses all six programming foundations?"
   - "How would I implement a simple database-like system with CRUD operations?"

---

## 🧠 Meta-Prompts About Prompting

These help students improve their prompting skills:

1. "If I wanted to learn about [concept], what would be the most effective way to prompt you?"
2. "Which parts of my prompt were clear, and which parts could be improved?"

---
*This document was co-authored with Claude, an AI assistant by Anthropic, to demonstrate effective human-AI collaboration in instructional design.*