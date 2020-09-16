---
layout: post
title: "Creating a R Notebook for Probability and Statistical Inference"
date: 2020-08-10
---
![](/images/carlos-muza-hpjSkU2UYSU-unsplash.jpg)

### Understanding variables and relationships between them
In building a General Linear Statistical Model (like Linear Regression model), we first need to understand and define accurately what is happening inside the data. 
This post is written exactly to find that using statistical tools. The post finds out if there are evidences of relationships in the data and it uses appropriate statistical tests and makes decisions based on their results. 
A relationship between variables is need to have a strength value and a direction associated with it. This post identifies if there are any differential effects for different groups using statistical tests like Anova and t-test.
The analysis is based on [Student Performance dataset](https://archive.ics.uci.edu/ml/datasets/student+performance) from UCI ML repository . The R notebook explores two relationships, one between number of absences and final maths grades, other one between maths grades in period one and final maths grades. It will also check for differential effects of gender on number of absences, mother’s job on a student’s final maths grades. Lastly, it will explore if maths grades in period one and two are similar or not.

In order to find out the relationships and differences, this report conducts statistical analysis on the given dataset, and the R Notebook can be accessed by clicking [here](https://github.com/yvrjsharma/R/blob/master/stats.md). For description of the variables present in this dataset, one can also refer this [paper](https://repositorium.sdum.uminho.pt/bitstream/1822/8024/1/student.pdf).
