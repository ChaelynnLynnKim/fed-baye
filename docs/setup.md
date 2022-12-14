# Fed-Baye Development Environment Setup

There are a number of dependency considerations to take into account in this project, in part because of the TensorFlow packages involved and in part because Google Cloud Platform's Vertex AI services, the main development platform here, generally run Python 3.7. To that end, the following serves as a Setup guide for configuring a Vertex AI notebook and effectively installing the packages in `requirements.txt`. 

## Vertex AI Notebook

First, log in to Google Cloud Platform and in your current project, enter <em>Vertex AI</em> in the search bar. From the <em>Tools</em> section on the left side of the screen, select <em>Workbench</em>, which will take you to the following screen.

<img src="./assets/Vertex AI.png"
     alt="Vertex AI Page"
     style="float: left; margin-right: 10px;" />

From this screen, click on <em>NEW NOTEBOOK</em> at the top of the page, and from the dropdown that appears, select <em>TensorFlow Enterprise</em> &rarr; <em>TensorFlow Enterprise 2.10</em> rarr; <em>With 1 NVIDIA T4</em>. When the next pop-up appears, select <em>Advanced Options</em> at the bottom of the screen.

<img src="./assets/TF Enterprise.png"
     alt="TF Enterprise Screen"
     style="float: left; margin-right: 10px;" />

In the <em>Machine Configuration</em> section, change the GPU from a NVIDIA T4 to the more powerful NVIDIA P100. Also ensure that the box next to "Install NVIDIA GPU driver automatically for me" is checked. Once complete, scroll to the bottom of the page, and click <em>Create</em>.

<img src="./assets/Machine Configuration.png"
     alt="Machine Configuration Pop-up"
     style="float: left; margin-right: 10px;" />

Once the Jupyter Notebook instance has been launched, click the <em>OPEN JUPYTERLAB</em> button to launch the Jupyter instance.


## Environment Configuration

After opening the Jupyter Notebook instance, open up a terminal from the <em>Launcher</em> screen and run the following commands:

```
$ sudo apt update
$ sudo apt install python3-dev python3-pip
$ python3 -m venv "fed-baye"
$ source "fed-baye/bin/activate"
(fed-baye) $ pip install --upgrade "pip"
```

This first set of commands is responsible for setting up the virtual environment that will be used to run the scrips and notebooks used in this project. Next, the project should be cloned into a new folder in the Jupyter Notebook instance, which can be accomplished by running `git clone https://github.com/k-bartolomeo/fed-baye.git`. You will be prompted for your GitHub username, as well as your [Personal Access Token](https://docs.github.com/en/enterprise-server@3.4/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token).

This command will create a local copy of the `fed-baye` repository on your Jupyter Notebook instance; this repository contains a `requirements.txt` file that will be used to install the project's dependencies. Navigate to the `fed-baye` directory from the current directory with `(fed-baye) $ cd fed-baye`, and then run `(fed-baye) $ pip3 install -r requirements.txt`. Once complete, your `fed-baye` virtual environment will contain the necessary packages needed to run this project's contents.

## Running a New Notebook

The last step is to ensure that the notebook you are running points to the virtual environment in which the project's dependencies were installed. This can be accomplished by running `(fed-baye) $ python3 -m ipykernel install --user --name=fed-baye`. This command will add the `fed-baye` environment to the list of kernel Jupyter Notebook can source. Return to the <em>Launcher</em> page, and under <em>Notebook</em>, you should see an option to start a notebook with `fed-baye (Local)`. Click this option to launch a Jupyter Notebook pointing to the `fed-baye` environment, and complete the development environment setup. 