---
title: "🎯 Intentional Prompting with AI: A Python Developer's Guide"
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



> Using AI isn't just about asking questions — it's about **guiding** the assistant to help you think, code, and reflect like a Python developer.

---

## 🧠 What Is Intentional Prompting?

**Intentional prompting** means you:

- Ask the AI to **explain** its reasoning
- Ask AI questions that lead to deeper understanding of Python concepts
- Refine and adapt prompts when the first response isn't quite right
- Know when to question, correct, or reject AI-generated Python code
- Use the AI as a **thinking partner**, not just a code vending machine

---

## 💬 Smart Python Prompts to Try

When working with AI tools like Claude, don't just say "Write the code." Try these:

| Goal | Prompt |
|------|--------|
| Understand the logic | `"Can you explain why you chose a list comprehension instead of a for loop?"` |
| Check assumptions | `"What assumptions are you making about the input data type?"` |
| Improve performance | `"Would a generator be more memory-efficient here than a list?"` |
| Think practically | `"How would this function handle large datasets in a real-world scenario?"` |
| Refine or improve | `"Can you show a more Pythonic version of this function?"` |
| Compare options | `"Can you show both a recursive and an iterative approach to this problem?"` |

Even if you don't copy the answer directly, these prompts help you **understand the reasoning** behind the Python code — and make you a stronger developer.

---

## 🧩 Examples of Intentional Prompting

### ❌ Lazy Prompt:
> "Write a function to sort a list."

**AI might give you:** a basic implementation that uses Python's built-in sort methods without explaining why or considering edge cases.

### ✅ Better Prompt:
> "Write a Python function to sort a list of dictionaries by a specific key, and explain how it handles missing keys."

Here, you're guiding the AI toward **specific, more thoughtful Python code**.

---

## 🧠 What to Do When AI Gets Python Wrong

Even a good prompt can return:
- Outdated Python practices (like Python 2.x syntax)
- Inefficient algorithms or data structures
- Logic errors or edge case issues

> Your job isn't just to spot errors — it's to **lead the AI** back to a better Python solution.

Try:

- `"This uses too many nested loops. Can you optimise it using Python's built-in functions?"`
- `"Can you rewrite this to handle None values properly?"`
- `"Your recursion has no base case. Can you add proper termination?"`

---

## 🔄 Prompting Is Iterative

Good prompting is **a conversation**, not a one-shot.

Start simple → evaluate → refine → test → ask again.  
Just like debugging your Python code.

---

## 🛠 Use This When…

- You're stuck on a Python algorithm and need a next step
- The AI gives "almost right" Python code
- You're prepping for a task that involves Python development or code review

---

## 🧳 Final Tip: Keep a Python Prompting Journal

When working on Python projects, keep a short log of:

- Prompts you tried
- What Python patterns worked and what didn't
- Your best "fix-it" prompt and how it improved the code

This shows **how you think** about Python, not just what you copied — and that's what makes you a better developer.

---