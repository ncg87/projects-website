class ImageCaptioning{
    constructor(){
        this.inputImage = document.getElementById('input');
        this.captionButton = document.getElementById('caption-button');
        this.imageContainer = document.querySelector('.image-container');
        this.imageField = document.querySelector('img');
        this.captionElement = this.imageContainer.querySelector('.caption');
    }
    // Handles clicking on the captioning button
    initialize() {
        console.log('Button Initalized');
        // Creates a listener for the button
        this.captionButton.addEventListener('click', () => this.onCaptionButton());
    }

    // Handles the captioning
    onCaptionButton() { 
        // Updates the caption element
        this.captionElement.innerHTML = 'Captioning...';
        // Obtains the file input by the user
        let inputFile = this.inputImage.files[0];
        // Checks if file is empty
        if(!inputFile) {
            alert("Choose a image to caption");
            console.log('No file selected');
            return;
        }
        // Packages the file for the captioning function
        let file = new FormData();
        file.append('image', inputFile);

        // Calls the caption function
        fetch($SCRIPT_ROOT + '/caption',{
            method: 'POST',
            body: file,
        })
        // Waits for the response and extracts the caption 
        .then(response => response.json())
        .then(response => {
            console.log('Caption response:', response.caption);
            // Creates URL of the image
            let fileURL = URL.createObjectURL(inputFile);
            // Updates the image and caption
            this.updateImageContainer(response.caption, fileURL);
        })
        .catch(error => {
            // Logs error
            console.error(`Error: ${error.message}`);
            // Hides image and caption on error
            this.imageContainer.style.display = 'none';
        });
    }

    updateImageContainer(caption, URL){
        // Displays the image container
        this.imageContainer.style.display = 'block';

        // Checks if the caption element is found
        if(this.captionElement){
            // Console logs that the caption element was found
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
// Creates an instance of the Captioning class
const captioning = new ImageCaptioning();

// Initalizes the button
captioning.initialize();