$(document).ready(function (e) {
    $('#upload').on('click', function () {
        $("#prediction").show()
        $("#msg").html("Loading...")
        var fileData = $('#image').prop('files')[0];

        var formData = new FormData();
        formData.append('file', fileData);
        $.ajax({
            url: 'http://localhost:5000/predict', // point to server-side controller method
            dataType: 'text', // what to expect back from the server
            cache: false,
            contentType: false,
            processData: false,
            data: formData,
            type: 'post',
            success: function (response) {
                response = JSON.parse(response)
                var message = "The predicted label is: " + response['label']
                $('#msg').html(message); // display success response from the server
                $('#response-form-container').show()
            },
            error: function (response) {
                $('#msg').html(response); // display error response from the server
            }
        });
    });

    $('#response-form').submit(function (e) {
        e.preventDefault()

        var correct = $('input[name="correct"]:checked').val();
        var userLabel = $('#response-label').val()
        var fileData = $('#image').prop('files')[0];

        var formData = new FormData();
        formData.append('file', fileData)
        console.log(formData)
        $.ajax({
            url: `http://localhost:5000/upload?correct=${correct}&label=${userLabel}`, // point to server-side controller method
            dataType: 'text', // what to expect back from the server
            cache: false,
            contentType: false,
            processData: false,
            data: formData,
            type: 'post',
            success: function (response) {
                response = JSON.parse(response)
            },
            error: function (response) {
                $('#msg').html(response); // display error response from the server
            }
        });
    });
});