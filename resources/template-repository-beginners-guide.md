---
title: "GitHub Template Repository: A Beginner’s Guide"
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

Welcome to the project! This handout explains how the GitHub template repository works, what happens after you create a repository from it, and some important points about forking versus using the template button. Please read it carefully so you know what to expect and how to keep your project up to date.

---

## 1. How Template Repositories Work

- **Snapshot Copy:**  
  When you use the **“Use template”** button on GitHub, you create a **new repository** that is a copy of the template repository. This new repository is a snapshot taken at that moment.  
- **No Ongoing Link:**  
  After you create your repository from the template, it is completely independent. Any future changes made to the template repository (by the instructor or others) will **not** automatically update your repository.

---

## 2. Keeping Your Repository Up to Date

- **Manual Updates Only:**  
  If the template is updated later (for example, improved documentation or bug fixes), these changes do not push themselves to your repository automatically.
- **What You Can Do:**  
  - **Manually Add Updates:** You may need to manually review the changes in the template and then update your repository by copying over the new content.
  - **Forking (Not Recommended for Beginners):**  
    While forking a repository creates an associated copy, it also comes with complications:  
      - If you fork a **public** repository, your fork remains public even if you want it private.
      - Forks are more useful when you plan to actively contribute back to the original project, but for this course, we recommend using the “Use template” button.

---

## 3. Why Not to Fork for This Project

- **Public Visibility:**  
  If you fork a public repository, your fork will automatically be public. In our case, we want your project to remain private.
- **No Built-in Update Mechanism:**  
  Although forks have a link to the original repository (called the upstream), the default workflow in our class is to use the template, which avoids additional steps like setting up upstream remotes.

---

## 4. Working Through the Web Interface

- **Web-Based Workflow:**  
  Since most of us use the GitHub web interface (and even link notebooks to GitHub with “Open in Colab” buttons), you won’t need to use the command line.
- **Creating Your Repository:**  
  - Click the **“Use template”** button in the template repository.
  - Name your repository and make sure you keep it **private**.
  - A basic `README` will be created automatically to ensure your repository exists.
- **When You Need Updates:**  
  When you see an announcement about updates to the template, review the changes and ask for assistance if you need help manually adding the updates to your project.

---

## 5. Where to Find This Information

- **On the LMS:**  
  This handout has been posted on the LMS where you can easily refer to it.
- **In the Template Repository:**  
  It’s a good idea for the instructor to include a copy of this handout in the template repository (for example, as a `GUIDE.md` file). This way, if you ever need a reminder of how things work, you can review it directly there.

---

## Summary

- Repositories created from the **“Use template”** button are independent copies.  
- There is **no automatic link** between your project and the original template repository.
- **Forking** a public repository is not recommended because the fork will remain public and require extra steps.
- Check the LMS and/or the template repository for any updated instructions.

If you have any questions or need help merging manual updates in the future, please contact your instructor or teaching assistant.

---

This handout is intended to give you a clear understanding of how our GitHub projects are set up and how to work with them. Enjoy coding and learning with your projects!