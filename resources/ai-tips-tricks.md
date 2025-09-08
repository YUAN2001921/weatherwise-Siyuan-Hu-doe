---
title: "🧠 AI Tips and Tricks for the Weather Advisor Project"
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

> This guide provides creative approaches to working with AI assistants that go beyond basic prompting. These techniques can help you develop a more sophisticated project while demonstrating your ability to use AI tools effectively as part of your development process.

## Using Multiple AI Assistants

### AI Debate Technique
Have two different AI assistants critique each other's solutions:

1. **Get an initial solution** from Assistant A
2. **Ask Assistant B to critique it**, highlighting strengths and weaknesses
3. **Take Assistant B's critique back to Assistant A** for improvements
4. **Document this "debate"** as part of your development process

Example prompt for the critique phase:
```
I received this code suggestion from another AI assistant for parsing weather questions. Can you analyse it and identify:
1. Potential edge cases it might miss
2. Performance considerations
3. Ways to make it more robust
4. Alternative approaches that might work better

[Paste code here]
```

### Different AIs for Different Roles

Assign different AI assistants to specialised roles:

- **Architect AI**: Help with overall design and structure
- **Implementation AI**: Generate specific code solutions
- **Testing AI**: Identify potential bugs and edge cases
- **Documentation AI**: Help document your code and process

Document how different AIs approached the same problem differently, and how you integrated their various perspectives.

## Advanced Prompting Techniques

### Chain-of-Thought Programming
Guide the AI through a programming problem step by step:

1. **Start with problem definition**: "I need to parse weather questions like 'Will it rain tomorrow in Sydney?'"
2. **Ask for an analysis**: "What are the key components we need to extract?"
3. **Request an approach**: "What techniques could we use to extract these components?"
4. **Get implementation**: "Can you implement the most suitable approach?"
5. **Ask for testing**: "How would you test this with various inputs?"

This shows a methodical development process rather than just asking for a complete solution.

### Reverse Engineering
Ask AI to explain existing code to build your understanding:

1. Find relevant code examples (e.g., weather API usage)
2. Ask the AI to explain how it works line-by-line
3. Ask what could be improved or customised
4. Use this understanding to build your own implementation

Example:
```
I found this code for working with the wttr.in API. Can you explain how it works line by line, and suggest how I might adapt it for my needs?

[Code example]
```

### Progressive Disclosure
Start with minimal information and gradually provide more context:

1. Begin with a basic problem statement: "How would you parse weather-related questions?"
2. After receiving initial advice, add constraints: "The solution needs to handle questions about time periods"
3. Then add more requirements: "It should also extract locations and weather attributes"
4. Finally, provide specific examples: "Like 'Will it rain tomorrow in Sydney?'"

This helps you understand how adding context affects solutions and prevents the AI from making too many assumptions.

### Persona-Based Prompting
Ask the AI to adopt different perspectives:

- "As a UX designer, how would you organise the weather dashboard interface?"
- "As a data scientist, what visualisations would be most informative for weather data?"
- "As a security expert, what should I consider when handling location data from users?"

Document how these different perspectives influenced your design decisions.

### Prompt Engineering Experiments
Try the same basic question with different prompting techniques:

1. Direct question: "How do I parse weather questions?"
2. Contextual question: "I'm building a weather app and need to parse questions. How should I approach this?"
3. Role-based: "You are an expert in NLP. How would you parse weather questions?"
4. Comparative: "What are three different approaches to parsing weather questions?"

Document how different prompting styles yield different quality responses.

## Creative Documentation Approaches

### Code Storytelling
Have the AI help narrate the development journey:

```
I've implemented the weather data retrieval function and visualisation components. Can you help me create a narrative that explains:
1. The challenges I encountered
2. The design decisions I made
3. How the components work together
4. What I learned in the process
```

This creates more engaging documentation that shows your learning process.

### Comparative Analysis
Ask the AI to compare multiple potential solutions:

```
I'm considering these three approaches for the user interface:
1. Console-based menu with pyinputplus
2. Interactive widgets with ipywidgets
3. A hybrid approach

Can you analyse the pros and cons of each for this specific project?
```

Document how this analysis informed your implementation choice.

### Code Review Practice
Have AI review your code, then critically evaluate its suggestions:

1. Share a function you've written with the AI
2. Ask for a detailed code review: "What could be improved in this code?"
3. Evaluate each suggestion critically - accept some, question others
4. Document your reasoning for accepting or rejecting each suggestion
5. Implement the changes you agree with

This demonstrates your ability to think critically about AI advice rather than accepting it blindly.

## Interactive Development

### Iterative Refinement Sessions
Document a series of improvements to a single component:

1. Start with a basic implementation
2. Ask: "What's one thing I could improve about this code?"
3. Implement the suggestion
4. Repeat several times, documenting each iteration
5. Reflect on how the code evolved through this process

This demonstrates incremental development rather than expecting perfect code immediately.

### Error-Driven Development
Deliberately introduce hypothetical errors to explore robustness:

```
What if a user enters 'near Sydney' instead of just 'Sydney'? How would our location parser handle this?

What would happen if the weather API returns incomplete data? How should our code handle that?
```

Document how these "what if" scenarios helped you build more robust code.

### AI-Assisted Testing
Use AI to generate comprehensive test cases:

1. Ask for a variety of test inputs: "What test cases should I use to verify my weather parsing function?"
2. Request edge cases: "What unusual inputs might break this function?"
3. Generate test data: "Can you provide sample weather data that would test all branches of my display function?"
4. Create a test plan: "How would you structure a testing approach for this entire application?"

Document how these AI-suggested tests improved your code's reliability.

### Alternative Implementation Comparison
Have AI generate multiple approaches to the same problem:

```
Can you show me three different ways to implement a function that parses weather questions, using:
1. Regular expressions
2. Natural language processing techniques
3. Simple string operations

What are the tradeoffs between these approaches in terms of complexity, maintainability, and robustness?
```

Compare the solutions and document your reasoning for choosing one approach over others.

## Remember:
The quality of your AI interactions will be evaluated based on how effectively you direct and learn from the AI, not on how impressive the AI's responses are. Focus on demonstrating thoughtful engagement with the AI as a tool in your development process.

## Additional Resources

For specific prompts related to the six core programming foundations (Input, Output, Store, Calculate, Decisions, and Repeat), refer to the **Prompt Templates** document provided with this assignment. These templates offer structured ways to ask about specific programming concepts you'll need for your Weather Advisor project.

The prompt templates are organised by programming concept and include:
- Concept exploration prompts
- Implementation prompts
- Debugging prompts
- Challenge prompts

Remember to adapt these templates to your specific implementation needs rather than using them verbatim.

---

*Note: Including conversations from multiple AI assistants and documenting creative AI usage approaches can demonstrate your ability to think critically about AI tools and may result in a more comprehensive solution. However, remember that the fundamental requirement is showing intentional prompting and a clear development process, regardless of which techniques you employ.*
