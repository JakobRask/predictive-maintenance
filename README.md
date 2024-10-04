<h1>Predictive Maintenance - Data Scientist Project using Time Series Forecasting</h1>

<h2>Project Description</h2>
<p>This project focuses on building a predictive maintenance model to predict machine failures using time-series sensor data. The goal is to anticipate when machines will fail, enabling proactive maintenance to minimize downtime and reduce costs. By leveraging historical sensor readings, error logs, failure events, and maintenance records, we aim to build a robust model capable of identifying early signs of failure.</p>

<h2>Dataset Summary</h2>
<p>The dataset used for this project contains sensor data, error logs, failure events, maintenance records, and machine metadata for 100 machines from 2014 to 2015. It is designed for Predictive Maintenance model building, providing insights into machine conditions and failure prediction.</p>

<h2>Key Questions We Aim to Solve</h2>
<ol>
    <li><strong>Can we predict anomalies in sensor readings (e.g., pressure, voltage) that indicate potential machine failure?</strong> 
        <p>We will train a time-series model (e.g., LSTM) on telemetry data to forecast sensor readings. By comparing the forecasted readings with historical failures, we aim to identify patterns that precede breakdowns.</p>
    </li>
    <li><strong>Can we accurately forecast machine failures before they occur using telemetry data and error logs?</strong> 
        <p>By analyzing sensor anomalies and their relation to failures, we will evaluate our model's capability to provide early warnings of machine breakdowns, which will be essential for scheduling timely maintenance.</p>
   
   measures to prevent failure based on a scheduled 
APP using streamlit that says:
1. "Your machine is likely to produce X error leading to Y failure due to high torque speed or vibration levels on this date w/ 95% confidence level". 
2. "The predicted failure date is Y date". "Change by A date for 85%, B date for 90%, C date for 95%".
3. "Carry out maintenance/service at latest by X date".
4. "The amount saved by carrying out service is $$". 
    </li>
</ol>

<h2>Project Objectives</h2>
<ul>
    <li>Develop and evaluate a time-series forecasting model using LSTM to predict machine sensor anomalies.</li>
    <li>Use predicted sensor readings to estimate the probability of machine failure.</li>
    <li>Explore potential improvements, including synthetic data generation, hyperparameter tuning, and signal smoothing, to enhance model performance.</li>
</ul>

