ml-projects
==============

Machine Learning projects using traditional algorithms.

## Naive Bayes Mini Project

A couple of years ago, J.K. Rowling (of Harry Potter fame) tried something interesting. She wrote a book, “The Cuckoo’s Calling,” under the name Robert Galbraith. The book received some good reviews, but no one paid much attention to it--until an anonymous tipster on Twitter said it was J.K. Rowling. The London Sunday Times enlisted two experts to compare the linguistic patterns of “Cuckoo” to Rowling’s “The Casual Vacancy,” as well as to books by several other authors. After the [results of their analysis](http://languagelog.ldc.upenn.edu/nll/?p=5315) pointed strongly toward Rowling as the author, the Times directly asked the publisher if they were the same person, and the publisher confirmed. The book exploded in popularity overnight.

I've done something very similar in this project. We have a set of emails, half of which were written by one person and the other half by another person at the same company. Our objective is to classify the emails as written by one person or the other based only on the text of the email. I start with Naive Bayes in this mini-project, and then expand in later projects to other algorithms.

I start by getting a list of strings. Each string is the text of an email, which has undergone some basic preprocessing; I then use the code to split the dataset into training and testing sets.

One particular feature of Naive Bayes is that it’s a good algorithm for working with text classification. When dealing with text, it’s very common to treat each unique word as a feature, and since the typical person’s vocabulary is many thousands of words, this makes for a large number of features. The relative simplicity of the algorithm and the independent features assumption of Naive Bayes make it a strong performer for classifying texts. In this mini-project, I downloaded and installed sklearn on my computer and used Naive Bayes to classify emails by author.

## Support Vector Machines Mini Project

In this mini-project, I tackle the exact same email author ID problem as the Naive Bayes mini-project, but now with an SVM. I clarify some of the practical differences between the two algorithms. I've also tuned parameters a lot more than in Naive Bayes. I compared the training and prediction times to Naive Bayes.

I concluded that Naive Bayes is great for text -- it’s faster and generally gives better performance than an SVM for this particular problem. Of course, there are plenty of other problems where an SVM might work better. Knowing which one to try when you’re tackling a problem for the first time is part of the art and science of machine learning. In addition to picking your algorithm, depending on which one you try, there are parameter tunes to worry about as well, and the possibility of overfitting (especially if you don’t have lots of training data).