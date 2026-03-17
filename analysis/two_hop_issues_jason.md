
len(df): 82020

len(result): 82020

Counterfactuals with e2_new generated: 81491

Number of rows wihtout e2_new: 529

Counterfactuals with e3_new generated: 65575

Number of rows without e3_new: 16445

## Passed stats

test_knowledge_consistency passed (same group same new knowledge)
0 edit rows with missing e3_new values

## Big Issue 1 - Overlap
e2_new overlap: 11120

e3_new overlap: 7415

## Big Issue 2 - edits with no propagated eval

Consecutive edit rows: 22408

## Big Issue 3 - Invalid knowledge????

test_valid_r2 = 0.468

Invalid rows (violating knowledge): 34579


## e3overlap example

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source_prompt</th>
      <th>e1_label</th>
      <th>e2_label</th>
      <th>e3_label</th>
      <th>e2_label_new</th>
      <th>e3_label_new</th>
      <th>row_role</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>the head of state of the country of citizenship of Stephen Harper is</td>
      <td>Stephen Harper</td>
      <td>Canada</td>
      <td>Charles III of the United Kingdom</td>
      <td>United States of America</td>
      <td>Charles III of the United Kingdom</td>
      <td>edit</td>
    </tr>
    <tr>
      <th>14</th>
      <td>the head of state of the country of citizenship of Bill Maher is</td>
      <td>Bill Maher</td>
      <td>United States of America</td>
      <td>Joe Biden</td>
      <td>Canada</td>
      <td>Joe Biden</td>
      <td>edit</td>
    </tr>
    <tr>
      <th>21</th>
      <td>the country of citizenship of the father of George W. Bush is</td>
      <td>George W. Bush</td>
      <td>George H. W. Bush</td>
      <td>United States of America</td>
      <td>Eduardo Frei Montalva</td>
      <td>United States of America</td>
      <td>propagated_eval</td>
    </tr>
    <tr>
      <th>75</th>
      <td>the capital of the country of citizenship of Barack Obama is</td>
      <td>Barack Obama</td>
      <td>United States of America</td>
      <td>Washington, D.C.</td>
      <td>United States of America</td>
      <td>Washington, D.C.</td>
      <td>propagated_eval</td>
    </tr>
    <tr>
      <th>76</th>
      <td>the capital of the country of citizenship of Abraham Lincoln is</td>
      <td>Abraham Lincoln</td>
      <td>United States of America</td>
      <td>Washington, D.C.</td>
      <td>United States of America</td>
      <td>Washington, D.C.</td>
      <td>propagated_eval</td>
    </tr>
  </tbody>
</table>
</div>



## Invalid Knowledge example

Invalid rows (violating knowledge): 34579


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source_prompt</th>
      <th>e1_label</th>
      <th>e2_label</th>
      <th>e3_label</th>
      <th>e2_label_new</th>
      <th>e3_label_new</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>the head of state of the country of citizenship of Stephen Harper is</td>
      <td>Stephen Harper</td>
      <td>Canada</td>
      <td>Charles III of the United Kingdom</td>
      <td>United States of America</td>
      <td>Charles III of the United Kingdom</td>
    </tr>
    <tr>
      <th>14</th>
      <td>the head of state of the country of citizenship of Bill Maher is</td>
      <td>Bill Maher</td>
      <td>United States of America</td>
      <td>Joe Biden</td>
      <td>Canada</td>
      <td>Joe Biden</td>
    </tr>
    <tr>
      <th>20</th>
      <td>the spouse of the father of George W. Bush is</td>
      <td>George W. Bush</td>
      <td>George H. W. Bush</td>
      <td>Barbara Bush</td>
      <td>Eduardo Frei Montalva</td>
      <td>Jay Grdina</td>
    </tr>
    <tr>
      <th>21</th>
      <td>the country of citizenship of the father of George W. Bush is</td>
      <td>George W. Bush</td>
      <td>George H. W. Bush</td>
      <td>United States of America</td>
      <td>Eduardo Frei Montalva</td>
      <td>United States of America</td>
    </tr>
    <tr>
      <th>22</th>
      <td>the head of state of the country of citizenship of Diego Velázquez is</td>
      <td>Diego Velázquez</td>
      <td>Spain</td>
      <td>Felipe VI of Spain</td>
      <td>Denmark</td>
      <td>Isaac Herzog</td>
    </tr>
  </tbody>
</table>
</div>