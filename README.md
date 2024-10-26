# CS4125 - Software Analysis and Design
Group Project to implement an Email Classifier with Design Patterns.

## Objective
The goal of this project is to design and implement an email classifier application that leverages multiple design patterns to achieve a modular, scalable, and maintainable software architecture.
The project aims to help youunderstand and apply design patterns in a real-world scenario, enhancing your skills in object-oriented design and software development.

## Project Overview
In this project, you will build an email classifier application that can automatically categorize incoming emails into variouscategories (categories are provided in the datasetsprovidedduring the labs).
The classifier will use different machine-learningalgorithms and heuristic rules to make these classifications. To achieve a flexible and extendable design, you are required to implement as many design patterns as possible.

For the machine learning part, you can implement 5 new models other than the random forest provided in Lab 2. You may implement any algorithms of your choice.

Hint: If you use models from SKLearn library (like Hist_GB, SGD, Adaboosting, Voting, and Random Trees Embedding, please see the link below) you don’t need to code differently to accommodate them.
Your code should implement adesign pattern that allows you to select a model dynamically

## Key Requirements
1. Email Classification System:
   - The system should be able to classify emails into categories provided in the dataset.
     Please note you were provided the gallery_app  dataset. Remember we are working using Agile methodology, which means changing requirements from clients should be accommodating.
     Add the email data customers collected regarding their ‘purchasing’of services. A newdataset is provided on Brightspace in the project folder.
   - Implement different classification techniques and allow dynamic switching between them.
2. Design Pattern Implementation:
   - Use a minimum of **5 different design patterns** in the application.
   - **Justify the use of each design pattern** with an explanation of how it enhances the design and meets specific project requirements.
     **NOTE:** To accommodate design patterns, you need to change the requirements of the email classifier discussed during the lab sessions

## Authored By
- Milan Kovacs
- Italo da Silva
- Bayan Nezamabad
- Jacob Beck
