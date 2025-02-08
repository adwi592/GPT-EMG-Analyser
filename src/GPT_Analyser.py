import re
import subprocess
from openai import OpenAI
from colorama import Fore, Back, Style
import credentials  # Assuming this module contains your OpenAI API key


class EMGAnalyser:
    def __init__(self):
        # Initialize OpenAI client with the API key
        self.client = OpenAI(api_key=credentials.openai_key)

        self.model = "gpt-4o"

        self.instructions = (
            "Use the EMG dataset for developing motion decoding models. Col 1 is labels, and the "
            "other columns are the corresponding EMG data.\n\n"
        )
        self.question_history = []
        self.answer_history = []

        # Parameters
        self.temperature = 1
        self.max_tokens = 2048
        self.frequency_penalty = 0
        self.presence_penalty = 0
        self.top_p = 1

    def update_history(self, question=None, answer=None):
        """Update the conversation history."""
        if question is not None:
            self.question_history.append(question)
        if answer is not None:
            self.answer_history.append(answer)

    def get_response(self, new_question):
        """Get a response from the OpenAI Chat API."""
        # Build the messages
        messages = [{"role": "system", "content": self.instructions}]

        # Add previous questions and answers to the conversation
        for question, answer in zip(self.question_history, self.answer_history):
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})

        # Add the new question
        messages.append({"role": "user", "content": new_question})

        # Call the OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "text"},
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )

        # Extract and return the content
        if response.choices and hasattr(response.choices[0].message, "content"):
            return response.choices[0].message.content
        else:
            return "Error: No content returned in the response."

    def extract_code(self, response):
        """Extract code blocks from GPT response."""
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
        return code_blocks

    def execute_code(self, code):
        """Execute extracted code after user confirmation and handle errors using LLM feedback."""

        # Ask for user confirmation before executing the code
        print(Fore.YELLOW + "\nThe following code was detected:\n" + Style.RESET_ALL)
        print(Fore.WHITE + code + Style.RESET_ALL)

        user_input = input(
            Fore.YELLOW + "\nDo you want to execute this code? (yes/no): " + Style.RESET_ALL).strip().lower()
        if user_input != "yes":
            print(Fore.RED + "Code execution aborted." + Style.RESET_ALL)
            return "Execution skipped by user."

        try:
            # Write code to a temporary file
            with open("temp_code.py", "w") as f:
                f.write(code)

            # Execute the code and capture the output
            result = subprocess.run(
                ["python", "temp_code.py"],
                text=True,
                capture_output=True,
            )

            # Return the captured output or error
            if result.returncode == 0:
                print(Fore.GREEN + "Execution successful!" + Style.RESET_ALL)
                return result.stdout
            else:
                error_message = result.stderr
                print(Fore.RED + "Error detected in execution:" + Style.RESET_ALL)
                print(Fore.RED + error_message + Style.RESET_ALL)

                # Send error back to LLM for debugging
                debug_prompt = (
                    "The following Python code resulted in an error. Please identify the issue and suggest a corrected version:\n\n"
                    f"Code:\n```python\n{code}\n```\n\nError Message:\n{error_message}"
                )
                correction = self.get_response(debug_prompt)
                print(Fore.CYAN + "\nLLM Suggested Fix:" + Style.RESET_ALL)
                print(correction)

                return f"Execution failed:\n{error_message}\n\nLLM Suggestion:\n{correction}"

        except Exception as e:
            return f"Execution failed: {str(e)}"

    def converse(self):
        """Initiates a conversation with the user."""
        print(Fore.MAGENTA + Style.BRIGHT + "Please type end/End/END to terminate the conversation!\n")
        # print("Welcome to the EMG Analyser Chat Interface! (Type 'exit' to end the conversation)\n")
        while True:
            new_question = input(Fore.GREEN + Style.BRIGHT + "User: " + Style.RESET_ALL)
            if new_question.lower() == "exit":
                print("Exiting the conversation. Goodbye!")
                break

            try:
                response = self.get_response(new_question)
                print(Fore.CYAN + Style.BRIGHT + f"EMG Assistant: " + Style.RESET_ALL + response)
                self.update_history(question=new_question, answer=response)

                # Check for code blocks in the response
                code_blocks = self.extract_code(response)
                if code_blocks:
                    print("Code detected. Executing...")
                    for code in code_blocks:
                        execution_result = self.execute_code(code)
                        print(f"Execution Result:\n{execution_result}")
            except Exception as e:
                print(f"Error: {str(e)}")
                break


# Example usage:
if __name__ == "__main__":
    # Ensure your OpenAI API key is in the credentials module
    analyser = EMGAnalyser()
    analyser.converse()
