# RocketPy Hackathon
## _Git and GitHub Introduction_

This tutorial is meant to guide the hackathon participants to the basic functionalities of Git and GitHub that they are going to need while solving the Challenges.

## Git Installation

In order to work with RocketPy Hackathon repository, one must have Git installed in the machine. The install instructions are available at [the official Git page](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) for Linux, Windows and Mac.

Furthermore, specially if the participant is more comfortable with a GUI interface for operating with Git (instead of terminal), it is optional to download [GitHub Desktop](https://desktop.github.com/).

## GitHub Account Configuration

A GitHub account is required for submitting changes to the Hackathon Repository. If the participant does not have an account, it is possible to sign up [here](https://github.com/signup?source=login).

After successfully creating an account, it must be set up with the participant _username_ and _email address_, this can be done with the following terminal commands:

```sh
git config --global user.name "your_username"
git config --global user.email "your_email@example.com"
```

## Repository Workflow

### Cloning

First and foremost, the participant must open a terminal session and change the current directory to the one in which the Hackathon Repository will be downloaded. Then it is necessary to clone the Hackathon GitHub repository with the following commands:

```sh
cd <your_desired_path>
git clone https://github.com/RocketPy-Team/RocketPy-Hackathon-2022.git
```

### Basic Workflow

The basic workflow of interacting with a GitHub repository is through getting the most recent version of code (pull), making necessary changes (commits) and uploading these changes to the remote repository (pushes). For instance, the basic workflow for updating a README.md file from a repository would be similar to this:

```sh
# getting most recent version
git pull
# selecting all changed files to commit
git add .
# update the local repository with changes and 
# a descriptive commit message
git commit -m "Update README.md file"
# upload changes to the remote repository
git push
```

### Pull Requests

Pull Requests are a way to notify others of the changes made to the code, so that these changes may be revised by others before being merged into the final code. Pull Requests are located at the _Pull Request_ tab on the main page of the GitHub repository.

A new Pull Request can be made by specifying which changes should be merged into the repository, as illustrates the figure below:

![pull_request_button](https://docs.github.com/assets/cb-34915/images/help/pull_requests/choose-base-and-compare-branches.png)

## Forks and Branches

### Forks

A Fork is a copy of a GitHub repository that allows the user to make changes to a repository while keeping the original unchanged, with changes being updated to the original repository through pull requests. The participant will solve the Challenges and make contributions to a local Fork of the Hackathon GitHub repository, so that it is possible to keep the solutions organized and avoid conflicts that arise from multiple groups editing the same code. It is recommended to use GitHub environment to create a Fork, the participant has to click on the top right corner of the Hackathon GitHub page:

![fork_button](https://docs.github.com/assets/cb-23088/images/help/repository/fork_button.png)

### Branches
   
A Branch is a separate development path for the repository. They are specially useful for developing new features, testing changes or bug fixing while keeping the original code unchanged. The changes made in a branch can be merged into another branch through a Pull Request. It is recommended that the participant organize its own workflow with branches, since there are multiple Challenges that must be solved (sometimes working on more than one Challenge concurrently). Branches can be created from the terminal with the following command:

```sh
git checkout -b <new_branch_name>
```

After this command, a new branch will be created and the user is automatically directed to it. Furthermore, changing branches is done with the command ```git checkout <branch_name>``` (existing branches can be seen either on GitHub repository page or in terminal with the command ```git branch```).

![branches_button](https://docs.github.com/assets/cb-107867/images/help/branches/branches-overview-link.png)

## Issues and Challenges

The Hackathon Challenges are going to be released in the form of issues at the RocketPy Hackathon repository. They will be available after the Hackathon starts and can be accessed by clicking on the tab _issues_ as seen below:

![issues_button](https://docs.github.com/assets/cb-25896/images/help/repository/repo-tabs-issues.png)

After solving a Challenge, the participant will have to make a pull request to the main Hackathon repository.

## Further Information

Should the participant have any issues, RocketPy team is willing to help at a wide range of time-zones. Further information on Git/GitHub and GitHub Desktop are available from the official [GitHub Docs](https://docs.github.com/en). The documentation for the main topics covered in this tutorial are available in the table below:

| Topic | Documentation |
| ------ | ------ |
| Git Install and Set Up| [get-started/quickstart/set-up-git](https://docs.github.com/en/get-started/quickstart/set-up-git) |
| GitHub Desktop | [get-started/using-github/github-desktop](https://docs.github.com/en/get-started/using-github/github-desktop) |
| Commits | [creating-and-editing-commits/about-commits](https://docs.github.com/pt/pull-requests/committing-changes-to-your-project/creating-and-editing-commits/about-commits) |
| Forks | [get-started/quickstart/fork-a-repo](https://docs.github.com/en/get-started/quickstart/fork-a-repo) |
| Branches | [creating-and-deleting-branches-within-your-repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository) |
| Pull Request | [get-started/quickstart/github-flow](https://docs.github.com/en/get-started/quickstart/github-flow) |

