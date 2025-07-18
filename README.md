
# CSCS-USI HPC/Data Analytics Summer University 2025

This repository contains the materials used in the Summer University, including source code, lecture notes and slides.
Material will be added to the repository throughout the course, which will require that students either update their copy of the repository, or download/checkout a new copy of the repository.

## Announcements

We will be using Slack to post news and links relevant to the event: you should receive an invitation to join the Summer University Slack workspace at latest one week before the event.

## Schedule

### Group 1 - In-person
<img width="935" height="555" alt="image" src="https://github.com/user-attachments/assets/c9ef9744-30cd-4ed6-8ddc-5ecf1abd2f16" />

### Group 2 - Online
<img width="938" height="467" alt="image" src="https://github.com/user-attachments/assets/b5561273-32ea-44ab-8712-2a2448187943" />

## Link to materials

- [CUDA](./cuda) (Day 1, 2 & 3)
- [Python HPC](./python-hpc) (Day 4, 5 & 6)

## Obtaining a copy of this repository

### On your own computer

You will want to download the repository to your laptop to get all of the slides.
The best method is to use git, so that you can update the repository as more slides and material are added over the course of the event.
So, if you have git installed, use the same method as for Piz Daint below (in a directory of your choosing).

You can also download the code as a zip file by clicking on the "<> Code" (  <img width="121" height="39" alt="image" src="https://github.com/user-attachments/assets/0a55224e-5ac9-4027-80be-22066a86073f" /> ) button on the top right hand side of the github page, then clicking on __Download zip__.

### On Daint@Alps via JupyterLab

- Go to https://jupyter-daint.cscs.ch/ and sign in using your CSCS course credentials 
- Launch JupyterLab (might take a couple of minutes)
  - Advanced reservation 'summer_uni' 
  - Default values for the other fields (unless told otherwise by the instructor)
- Launch a new terminal : File -> New -> Terminal
- Issue the following commands on the terminal:
```bash
ln -s $SCRATCH scratch
cd $SCRATCH
git clone https://github.com/eth-cscs/SummerUniversity.git
```

### On Daint@Alps via ssh

This is an alternative method to the JupyterLab method above

```bash
# log onto Piz Daint ...
ssh classNNN@ela.cscs.ch
ssh daint

# go to scratch
cd $SCRATCH
git clone https://github.com/eth-cscs/SummerUniversity.git
```

## Updating the repository

Lecture slides and source code will be regurlarly updated on the remote git repository throughout the course.
To update your local repository you can simply go inside the path and type

```
git pull origin main
```

There is a posibility that you might have a conflict between your working version of the repository and the origin.
In this case you can ask one of the assistants for help.

# How to access Daint@Alps

This will be covered in the lectures and you can find more details in the [CSCS User Documentation](https://docs.cscs.ch/clusters/daint/#daint).
