<h1>Data Dictionary</h1>

<h2>1. Telemetry Data (PdM_telemetry.csv)</h2>
<p>This file contains hourly sensor data from 100 machines for the year 2015.</p>
<table>
  <thead>
    <tr>
      <th>Column Name</th>
      <th>Data Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>datetime</td>
      <td>datetime</td>
      <td>Date and time of sensor reading (hourly)</td>
    </tr>
    <tr>
      <td>machineID</td>
      <td>integer</td>
      <td>Unique ID for each machine</td>
    </tr>
    <tr>
      <td>voltage</td>
      <td>float</td>
      <td>Average voltage sensor reading</td>
    </tr>
    <tr>
      <td>rotation</td>
      <td>float</td>
      <td>Average rotation sensor reading</td>
    </tr>
    <tr>
      <td>pressure</td>
      <td>float</td>
      <td>Average pressure sensor reading</td>
    </tr>
    <tr>
      <td>vibration</td>
      <td>float</td>
      <td>Average vibration sensor reading</td>
    </tr>
  </tbody>
</table>

<h2>2. Error Data (PdM_errors.csv)</h2>
<p>This file records non-failure errors encountered by machines during operation, rounded to the nearest hour.</p>
<table>
  <thead>
    <tr>
      <th>Column Name</th>
      <th>Data Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>datetime</td>
      <td>datetime</td>
      <td>Date and time of the error event (rounded to the nearest hour)</td>
    </tr>
    <tr>
      <td>machineID</td>
      <td>integer</td>
      <td>Unique ID for each machine</td>
    </tr>
    <tr>
      <td>errorID</td>
      <td>integer</td>
      <td>Error code that identifies the type of error encountered</td>
    </tr>
  </tbody>
</table>

<h2>3. Maintenance Data (PdM_maint.csv)</h2>
<p>This file records maintenance activities on machines, including both proactive and reactive maintenance.</p>
<table>
  <thead>
    <tr>
      <th>Column Name</th>
      <th>Data Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>datetime</td>
      <td>datetime</td>
      <td>Date and time of the maintenance activity (rounded to the nearest hour)</td>
    </tr>
    <tr>
      <td>machineID</td>
      <td>integer</td>
      <td>Unique ID for each machine</td>
    </tr>
    <tr>
      <td>comp</td>
      <td>string</td>
      <td>Component that was replaced or maintained</td>
    </tr>
  </tbody>
</table>

<h2>4. Failure Data (PdM_failures.csv)</h2>
<p>This file records instances of machine component failure, requiring a component replacement.</p>
<table>
  <thead>
    <tr>
      <th>Column Name</th>
      <th>Data Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>datetime</td>
      <td>datetime</td>
      <td>Date and time of the failure event (rounded to the nearest hour)</td>
    </tr>
    <tr>
      <td>machineID</td>
      <td>integer</td>
      <td>Unique ID for each machine</td>
    </tr>
    <tr>
      <td>comp</td>
      <td>string</td>
      <td>Component that failed and needed replacement</td>
    </tr>
  </tbody>
</table>

<h2>5. Machine Metadata (PdM_machines.csv)</h2>
<p>This file contains meta-information about each machine, such as model type and the number of years in service.</p>
<table>
  <thead>
    <tr>
      <th>Column Name</th>
      <th>Data Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>machineID</td>
      <td>integer</td>
      <td>Unique ID for each machine</td>
    </tr>
    <tr>
      <td>model</td>
      <td>string</td>
      <td>Model type of the machine</td>
    </tr>
    <tr>
      <td>age</td>
      <td>integer</td>
      <td>Age of the machine in years (years in service)</td>
    </tr>
  </tbody>
</table>
