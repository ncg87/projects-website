class UploadAndRecieve{
    constructor(){
        this.inputImage = document.getElementById('input');
        this.captionButton = document.getElementById('caption_button');
        this.imageContainer = document.querySelector('.image-container');
        this.imageField = document.querySelector('img');
        this.captionElement = this.imageContainer.querySelector('.caption');
    }
    // Handles clicking on the captioning button
    initialize() {
        console.log('Button Initalized');
        // Creates a listener for the button
        this.captionButton.addEventListener('click', () => this.onButton());
    }

    // Handles the captioning
    onButton() { 
        this.captionElement.innerHTML = 'Captioning...';
        // Obtains the file input by the user
        let inputFile = this.inputImage.files[0];
        let fileURL = URL.createObjectURL(inputFile);
        // Checks if file is empty
        if(!inputFile) {
            alert("Choose a image to caption");
            console.log('No file selected');
            return;
        }
        // Packages the file for the captioning function
        let file = new FormData();
        file.append('image', inputFile);    

        // Calls the file function in Flask
        fetch('http://127.0.0.1:5000//file',{
            method: 'POST',
            body: file,
        })
        // Waits for the response and extracts the caption 
        .then(response => response.json())
        .then(response => {
            console.log('Caption response:', response);
            //var caption = response.caption;
            // Updates the caption
            this.updateImageContainer(response.caption, fileURL);
            // Shows image and caption 
            
        }).catch(error => {
            alert(error);
            console.log('Error:', error);
            // Hides image and caption
            this.imageContainer.style.display = 'none';
        });
    }

    updateImageContainer(caption, URL) {
        // Displays the image container
        this.imageContainer.style.display = 'block';
        
        // Checks if the caption element is found
        if(this.captionElement){
            console.log('Caption element found inside imageContainer.');

            // Updates the caption
            this.captionElement.innerHTML = caption;
            console.log('Caption updated');
            // Updates the image
            this.imageField.src = URL;
        } else {
            console.error('Caption element not found inside imageContainer.');
        }
    }
}
// Creates an instance of the Translator class
const test = new UploadAndRecieve();

// Initalizes the button
test.initialize();