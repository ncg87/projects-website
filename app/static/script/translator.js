class Translator {
    constructor() {
        this.translatorButton = document.querySelector('.translate_button');
        this.inputField = document.querySelector('input');
        this.translationResult = document.getElementById('translation_result');
    }

    // Handles listening for translation
    display() {
        // Creates a listener for the button
        this.translatorButton.addEventListener('click', () => this.onTranslateButton());
        // Creates a listener for when enter is pressed in the input field
        this.inputField.addEventListener('keyup', (e) => {
            if (e.key === 'Enter') {
                this.onTranslateButton();
            }
        });
    }

    // Handles the translation
    onTranslateButton() {
        console.log('Button Clicked');
        // Obtains the input typed by the user
        let text = this.inputField.value
        // Checks if the input is empty
        if(text === ""){
            return;
        }
        // Obtains the target language
        let target_language = document.getElementById("target_language").value;
        let source_language = document.getElementById("source_language").value;
        // Checks if the target language is the same as the source language
        if(target_language === source_language){
            this.updateTranslator(text);
            return;
        }

        //Packages the data for the translate function
        let data = {
            'text': text,
            'target_language': target_language
        }
        // Calls the translate function
        fetch($SCRIPT_ROOT + '/predict',{
            method: 'POST',
            body: JSON.stringify(data),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
        // Waits for the response and extracts the translation
        .then(response => response.json())
        .then(response => {
            console.log('Translation response:', response);
            // Updates the translation
            this.updateTranslator(response.translation);
            this.inputField.value = '';
            
        }).catch(error => {
            console.error('Error:', error);
            this.updateTranslator("Error");
            this.inputField.value = '';
        });
    }

    updateTranslator(translator){
        // Updates the translation
        this.translationResult.innerHTML = translator;
    }

}

// Creates a new translator object
const translator = new Translator();
// Shows the translator/activates listeners
translator.display();
