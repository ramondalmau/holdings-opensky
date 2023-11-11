# Self-Supervised Learning for Characterizing Airborne Holdings :question: :crystal_ball:

In this repository, we provide the Jupyter notebook and dataset necessary to replicate the results from our paper:

```
@inproceedings{Dalmau2023,
  author={Dalmau, Ramon and Very, Philippe and Jarry, Gabriel},
  booktitle={Proceedings of the 11th OpenSky Symposium}, 
  title={On the Causes and Environmental Impact of Airborne Holdings at Major European Airports}, 
  year={2023},
  pages={1--12},
  month={October},
  address={Toulouse, France}
}
```

## Files

The key resource in this repository is the [dataset.csv](dataset.csv) file ([doi:10.5281/zenodo.10032730](10.5281/zenodo.10032730)), which contains the observations required to reproduce the self-training process outlined in the [holding-study.ipynb](holding-study.ipynb) notebook.

This dataset is the result of a meticulous process involving the detection of holdings using the neural network from the `traffic` tool, computation of fuel consumption using `OpenAP`, grouping holdings by airport and 30-minute intervals, and merging the results with meteorological aerodrome reports (METARs) processed with `metafora` and air traffic flow management (ATFM) regulations information. Please note that we've excluded the raw traffic, weather, and ATFM regulations data to protect confidentiality. Furthermore, we've removed the airport and time columns, as the primary objective of this paper was just to demonstrate the potential of the self-training method.

Rest assured, this README contains comprehensive guidance and code examples to assist you in generating the dataset :smile:.

## Installation

We recommend to use Python 3.10 :snake:, and to install the libraries in [requirements.txt](requirements.txt) as follows:

```
pip install -r requirements.txt
```

We strongly recommend initiating the process by creating a clean conda environment (with Python 3.10). Execute the provided command to install the required libraries. It's crucial to strictly adhere to this procedure, as we will not address issues that arise if these steps are not followed meticulously.

## For those starting from raw data ...

If you intend to detect the holdings, process the weather data, and compute fuel consumption, make sure to install the following additional libraries:

```
traffic==2.8.1
metafora==1.1.2
openap==1.3
```

### Holding detection

To extract the holding patterns from a Traffic instance, you can utilize the following code:

```python
from traffic.core import Traffic

def get_holdings(traffic: Traffic, 
				 icao: str, 
				 max_workers: int = 4, 
				 max_distance: int = 75, 
				 min_altitude: int = 0) -> Traffic:
    """Returns holdings

    Args:
        traffic (Traffic): traffic instance
		icao (str): ICAO code of the destination airport
		max_workers (int): maximum number of workers
		max_distance (int): parameter to filter by distance from the airport (in nm)
		min_altitude (int): parameter to filter by altitude (in ft)

    Returns:
        Traffic: holding patterns in the traffic instance
    """    

    holdings = (traffic.distance(airports[icao])
                .query(f"(distance < {max_distance}) & (altitude > {min_altitude})")
                .resample('1s')
                .all(f'holding_pattern()')
                .eval(max_workers=max_workers))

    return holdings
```

### Fuel flow estimation

Next, you can estimation the fuel flow for each second of a holding using `OpenAP`:

```python
holding = holding.fuelflow()
```

The output holding instance is the same, but it's enriched with three additional features: the mass, the fuel flow (in kg/s), and the total burnt fuel (in kg). Please note that this process is executed for each holding independently. You can then concatenate these enriched holdings to obtain the dataset with these three additional columns.

### Weather data processing

To process the weather observations (which you can obtain for free from [Aviation Weather Centre (AWC)](https://aviationweather.gov/) or [ogimet](https://www.ogimet.com/), for instance), you can use the open-source tool `metafora` and the following functions (remember to use the version specified in the `requirements.txt`, as a previous version of `metafora` was used in this study) :

```python
from metafora import Metar, ml_features
from metafora.enums import WeatherPrecipitation, WeatherObscuration, OtherWeather
from typing import Dict, List, Optional
from pandas import DataFrame, json_normalize, to_datetime


def process_metars(metars: List[Dict], errors: Optional[str] = "ignore") -> List[Dict]:
    """Processes list of METARs with metafora

    Args:
        metars (List[Dict]): list of raw METARs with associated times. Each element of the list consists of a dictionary with two keys: "time": the time of the observation, and "report": the textual METAR. 

    Returns:
        List[Dict]: list processed METARs
    """
    reports = []

    for m in metars:
        try:
            # parse METAR from raw text
            metar = Metar.from_text(m["report"])

            # extract ml features
            report = ml_features(metar)

            # set release time in isoformat
            report["time"] = m["time"]
            reports.append(report)
        except Exception:
            logger.error("Error processing {}".format(m["report"]))
            if errors == "raise":
                raise ValueError("{} is not a valid METAR".format(m["report"]))

    return reports
	
	
def metars_to_dataframe(reports: List[Dict]) -> DataFrame:
    """Converts list of processed METARs to pandas dataframe
    where each row is a report and each column a feature

    Args:
        List (Dict): list of reports processed using the process_metars function

    Returns:
        DataFrame: METARs as dataframe
    """
    # json normalize, back to pandas
    df = json_normalize(reports, sep="_")

    # convert time to datetime
    df["time"] = to_datetime(df["time"], errors="coerce")

    # remove invalid METARs
    df.dropna(subset=["station", "time"], inplace=True)

    # CAVOK status must be boolean
    if "visibility_cavok" in df.columns:
        df["visibility_cavok"] = df["visibility_cavok"].fillna(False)
    else:
        df["visibility_cavok"] = False

    # create simplified columns
    df = simplify_weather(df)

    return df


def simplify_weather(df: DataFrame) -> DataFrame:
    """Simplifies the weather dataframe

    Args:
        df (DataFrame): original weather dataframe

    Returns:
        DataFrame: simplified weather dataframe
    """
    df['precipitation'] = False
    df['obscuration'] = False
    df['other'] = False
    df['thunderstorms'] = False
    df['freezing'] = False
    df['snow'] = False

    for c in df.columns:
        if c.endswith("_phenomena"):
            # precipitation
            status = df[c].str.startswith(tuple(WeatherPrecipitation._member_names_)).fillna(False)
            df["precipitation"] = df["precipitation"] | status

            # snow
            status = df[c].str.contains("SN").fillna(False)
            df["snow"] = df["snow"] | status

            # obscuration
            status = df[c].str.startswith(tuple(WeatherObscuration._member_names_)).fillna(False)
            df["obscuration"] = df["obscuration"] | status

            # other phenomena
            status = df[c].str.startswith(tuple(OtherWeather._member_names_)).fillna(False)
            df["other"] = df["other"] | status
        elif c.endswith("_descriptor"):
            # thunderstorms
            status = (df[c] == "TS").fillna(False)
            df["thunderstorms"] = df["thunderstorms"] | status

            # freezing
            status = (df[c] == "FZ").fillna(False)
            df["freezing"] = df["freezing"] | status

    # clouds 
    if "clouds_cloud" in df.columns:
        df['clouds'] = df["clouds_cloud"].notnull()
    else:
        df["clouds_cloud"] = None 
        df["clouds"] = False 

    return df 
	
	
metars = [{"time": "...", "report": "..."}, ...]
processed_metars = process_metars(metars)
weather = metars_to_dataframe(processed_metars)
```

### Merging data

To merge holdings with weather, begin by grouping them by airport and 30-minute intervals:

```python
from pandas import Grouper 

X = holdings.data.rename({'destination': 'airport'}, axis=1).groupby(['airport', Grouper(key='timestamp', freq='30T', closed='left')]).fuelflow.agg(['sum', 'size'])
X.columns = ['fuel_consumption', 'holding_time']
X['holding_time'] = X['holding_time'] / 3600 # in hours 
X.reset_index(inplace=True)
```

and then use the `merge_asof` function of `pandas`. Make sure to only keep the desired columns from the weather `DataFrame`!

```python
from pandas import merge_asof
 
X = merge_asof(holdings.sort_values('timestamp'),
               weather.rename({'time': 'timestamp'}, axis=1).sort_values('timestamp'),
               on='timestamp',
               left_by='airport',
               right_by='station',
               direction="forward",
               tolerance=timedelta(hours=1)).dropna(subset=['station'])
```

The final step involves merging with ATFM regulations to add the necessary labels. However, I regret that I am unable to share this data due to confidentiality reasons. :pray: :sweat_smile: ... I apologize if this information is disappointing at this stage. Nevertheless, you may explore alternative methods for labeling observations. To assist you, I am providing an anonymized extract that serves two purposes: (1) to demonstrate the structure of the regulations data, and (2) to illustrate the process of merging them with the dataset for labeling observations. 

The anonymized extract with three columns: airport where the regulation was applied, start and end time of the regulation as well as reason:

|airport|start_time         |end_time           |reason|
|-------|-------------------|-------------------|------|
|A      |2023-10-01 04:00:00|2023-10-01 16:00:00|Other     |
|B      |2023-10-01 07:10:00|2023-10-01 08:05:00|Other          |
|B      |2023-10-01 09:40:00|2023-10-01 10:35:00|Other          |
|C      |2023-10-01 05:00:00|2023-10-01 06:40:00|Other               |
|D      |2023-10-01 04:00:00|2023-10-01 22:00:00|Other               |
|E|2023-10-01 16:00:00|2023-10-01 17:00:00|Other     |
|E|2023-10-01 16:00:00|2023-10-01 18:00:00|Other     |
|E|2023-10-01 12:00:00|2023-10-01 13:20:00|Other     |
|F|2023-10-01 09:00:00|2023-10-01 11:00:00|Other     |
|G|2023-10-01 18:00:00|2023-10-01 20:30:00|Other     |
|H|2023-10-01 21:00:00|2023-10-01 23:00:00|Other     |
|I|2023-10-01 12:00:00|2023-10-01 13:00:00|Other     |
|J|2023-10-01 16:00:00|2023-10-01 17:20:00|Other     |
|K|2023-10-01 17:30:00|2023-10-01 19:00:00|Other     |
|K |2023-10-01 19:40:00|2023-10-01 21:20:00|Other     |
|L|2023-10-01 07:00:00|2023-10-01 16:00:00|Other     |
|M|2023-10-01 05:00:00|2023-10-01 07:00:00|Other     |
|N|2023-10-01 05:40:00|2023-10-01 08:00:00|Other   |
|O|2023-10-01 03:30:00|2023-10-01 10:00:00|Weather|
|O|2023-10-01 18:40:00|2023-10-01 20:20:00|Weather     |
|P |2023-10-01 10:20:00|2023-10-01 14:00:00|Weather     |
|Q|2023-10-01 03:00:00|2023-10-01 06:00:00|Weather     |
|Q|2023-10-01 18:00:00|2023-10-01 21:00:00|Weather    |
|R|2023-10-01 18:00:00|2023-10-01 21:00:00|Weather         |
|S|2023-10-01 06:00:00|2023-10-01 09:00:00|Weather         |

And the code the merge these data with the dataset for labelling obdservations: 

```python
from pandas import merge
 
X = merge(X, regulations, on="airport", how="left")
mask = ~X["time"].between(X["start_time"], X["end_time"], inclusive="left")
X.loc[mask, "reason"] = None
X.sort_values(["reason"], na_position = "last", inplace=True)
X.drop_duplicates(subset=["airport", "time"], keep="first", inplace=True)
```


## Contribute :bug: 

Feel free to submit any problems, suggestions, or questions you may have, and we will do our best to address them promptly. Thank you for your contribution! :love_letter:
