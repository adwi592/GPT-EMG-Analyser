# ChatGPT-EMG-Analyser

## Overview
This repository provides a framework for automating feature extraction and model training for electromyography (EMG)-based motion decoding using the ChatGPT API. The goal is to compare LLM-generated feature extraction and model training code against manually written implementations.

## Demo Video
Watch the demo below to see the system in action:
![Demo GIF](media/demo.gif)  


## Repository Structure

```
/ChatGPT-EMG-Analyser
│── /src                    # Source code directory
│   ├── GPT_Analyser.py      # Handles ChatGPT API interactions and execution
│   ├── credentials.py       # Stores API key (DO NOT commit this file)
│── /data                   # (Optional) EMG dataset storage (gitignored)
│── /media                  # Video and image assets
│   ├── demo.mp4            # Full demo video
│   ├── demo.gif            # Preview GIF
│── .gitignore              # Ignore sensitive files
│── README.md               # Project description and setup instructions
│── requirements.txt        # Dependencies for the project
│── LICENSE                 # License file (if needed)
```

## Installation and Setup

### Clone the Repository
```
git clone https://github.com/adwi592/GPT-EMG-Analyser.git
cd GPT-EMG-Analyser
```

### Install Dependencies
```
pip install -r requirements.txt
```

### Configure API Key
- Open `credentials.py` and replace `"your_openai_api_key_here"` with your actual OpenAI API key.

## Running the Project

### Run the main script
```
python src/GPT_Analyser.py
```

## Security and Best Practices
- Do not commit `credentials.py` (API key file).
- Use `.gitignore` to exclude sensitive files.
- Follow ethical AI usage for generating code.

## License
This project is licensed under the MIT License.

## Contributing
Contributions are welcome.  
1. Fork the repository  
2. Create a new branch (`feature-branch`)  
3. Commit changes and submit a pull request  
