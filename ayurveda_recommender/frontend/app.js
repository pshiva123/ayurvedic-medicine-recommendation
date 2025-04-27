// list of symptoms must match your backend order exactly
const symptomFeatures = [
  "acidity","indigestion","headache","blurred_and_distorted_vision",
  "excessive_hunger","muscle_weakness","stiff_neck","swelling_joints",
  "movement_stiffness","depression","irritability","visual_disturbances",
  "painful_walking","abdominal_pain","nausea","vomiting",
  "blood_in_mucus","fatigue","fever","dehydration","loss_of_appetite",
  "cramping","blood_in_stool","gnawing","upper_abdomain_pain",
  "fullness_feeling","hiccups","abdominal_bloating","heartburn",
  "belching","burning_ache"
];

window.onload = () => {
  const form = document.getElementById("medicineForm");
  const outputBox = document.getElementById("output");

  // hide on click
  outputBox.onclick = () => {
    outputBox.style.display = "none";
  };

  form.onsubmit = async (e) => {
    e.preventDefault();

    // gather inputs
    const age = parseInt(document.getElementById("age").value, 10);
    const gender = document.getElementById("gender").value.toLowerCase();
    const severity = document.getElementById("severity").value.toUpperCase();

    // build symptom flags
    const symptoms = {};
    document.querySelectorAll("input[name='symptom']").forEach(input => {
      symptoms[input.value.toLowerCase()] = input.checked ? 1 : 0;
    });

    const payload = { age, gender, severity, symptoms };

    try {
      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      outputBox.innerText = data.error
        ? `❌ Error: ${data.error}`
        : `🧠 Predicted Disease: ${data.disease}\n💊 Recommended Medicine: ${data.medicine}`;
    } catch (err) {
      outputBox.innerText = `⚠️ Request failed: ${err.message}`;
    }

    outputBox.style.display = "block";
  };
};
