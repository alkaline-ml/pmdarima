<!-- Adapted from Atom's CONTRIBUTING.md file: https://github.com/atom/atom/blob/master/CONTRIBUTING.md -->
# Contributing to `pmdarima`

First off, thanks for taking the time to contribute!

The following is a set of guidelines for contributing to pmdarima. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

#### Table Of Contents

[Resources](#resources)

[How Can I Contribute?](#how-can-i-contribute)  
  * [Filing Issues](#filing-issues)  
    * [Filing A Bug](#filing-a-bug)  
    * [Filing A Feature Request](#filing-a-feature-request)  

  * [Contributing Code](#contributing-code)  
    * [First Time Contributor?](#first-time-contributor)  
    * [Developing Locally](#developing-locally)  
    * [Pull Requests](#pull-requests)

## Resources

[Official Documentation](https://www.alkaline-ml.com/pmdarima/)  
[Issue Tracker](https://github.com/alkaline-ml/pmdarima/issues)

## How Can I Contribute?

### Filing Issues

#### Filing A Bug

Fill out the [required template](https://github.com/alkaline-ml/pmdarima/issues/new?assignees=&labels=%3Abeetle%3A+%3A+bug&template=BUG_REPORT.md&title=), the information it asks us for helps us resolve issues faster

> **Note:** If you find a **Closed** issue that seems like it is the same thing that you're experiencing, open a new issue and include a link to the original issue in the body of your new one.

##### How Do I Submit A _Good_ Bug Report?

Explain the problem and include additional details to help maintainers reproduce the problem:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Describe the exact steps which reproduce the problem** in as many details as possible.
* **Provide specific examples to demonstrate the steps**. Include links to files or GitHub projects, or copy/pasteable snippets, which you use in those examples. If you're providing snippets in the issue, use [Markdown code blocks](https://help.github.com/articles/markdown-basics/#multiple-lines).
* **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
* **Explain which behavior you expected to see instead and why.**

#### Filing A Feature Request

Fill out the [required template](https://github.com/alkaline-ml/pmdarima/issues/new?assignees=&labels=&template=FEATURE_REQUEST.md&title=), the information it asks us for helps us resolve issues faster

### Contributing Code

There are only a handful of core contributors to the `pmdarima` project, so any help is appreciated! See our [official docs](https://www.alkaline-ml.com/pmdarima/contributing.html#how-to-contribute) for more detailed instructions on how to fork and clone the repo.

#### First Time contributor? 
Start by looking for the `help wanted` or `good first issue` tags to help you dive in.

#### Developing Locally
* To build `pmdarima` from source, you will require `cython>=0.29` and `gcc` (Mac/Linux) or `MinGW` (Windows).
* _Always_ use a feature branch
* Be sure to add tests for any new functionality you create
* Make sure your change doesn't break anything else by running the test suite using one of the following:

```bash
$ make test
```

or  

```bash
$ python setup.py develop
$ pytest
```

#### Pull Requests

The process described here has several goals:

* Maintain `pmdarima`'s quality
* Fix problems that are important to users
* Engage the community in working toward the best possible `pmdarima`
* Enable a sustainable system for `pmdarima`'s maintainers to review contributions

Please follow these steps to have your contribution considered by the maintainers:

1. Follow all instructions in [the template](PULL_REQUEST_TEMPLATE.md)
2. After you submit your pull request, verify that all [status checks](https://help.github.com/articles/about-status-checks/) are passing <details><summary>What if the status checks are failing?</summary>If a status check is failing, and you believe that the failure is unrelated to your change, please leave a comment on the pull request explaining why you believe the failure is unrelated. A maintainer will re-run the status check for you. If we conclude that the failure was a false positive, then we will open an issue to track that problem with our status check suite.</details>

While the prerequisites above must be satisfied prior to having your pull request reviewed, the reviewer(s) may ask you to complete additional design work, tests, or other changes before your pull request can be ultimately accepted.
