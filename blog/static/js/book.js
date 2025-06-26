$(document).ready(function() {
    $('#id_department').change(function() {
        var deptId = $(this).val();
        $.getJSON('{% url "get_doctors" %}', { department_id: deptId }, function(data) {
            var doctorSelect = $('#id_doctor');
            doctorSelect.empty();
            doctorSelect.append('<option value="" disabled selected>Select a doctor</option>');
            $.each(data.doctors, function(i, doc) {
                doctorSelect.append('<option value="' + doc.id + '">' + doc.name + '</option>');
            });
        });
        console.log("hi");
    });

    $('#id_doctor').change(function() {
      console.log("hi");
      var doctorId = $(this).val();
      var date = $('#id_date').val();
      if (doctorId && date) {
          $.getJSON('{% url "get_available_times" %}', { doctor_id: doctorId, date: date }, function(data) {
              var timeSelect = $('#id_time');
              timeSelect.empty();
              timeSelect.append('<option value="" disabled selected>Select a time</option>');
              $.each(data.available_times, function(i, time) {
                  timeSelect.append('<option value="' + time + '">' + time + '</option>');
              });
          });
      }
  });


});