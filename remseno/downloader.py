# https://github.com/planetlabs/planet-client-python/blob/main/examples/data_download_multiple_assets.py
# https://developers.planet.com/docs/data/psscene/
# ortho_analytic_8b
# https://github.com/planetlabs/planet-client-python/blob/main/examples/orders_create_and_download_multiple_orders.py

# Copyright 2022 Planet Labs PBC.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# """Example of downloading multiple assets in parallel
#
# This is an example of getting two assets, activating them, waiting for them to
# become active, downloading them, then validating the checksums of downloaded
# files.
#
# [Planet Explorer](https://www.planet.com/explorer/) was used to define
# the AOIs and get the image ids.
# """

# Need to download planetscope data for these regions namely, a 1000m2 region
import pandas as pd
from requests.auth import HTTPBasicAuth
import requests
import asyncio
import os
import planet

API_KEY = 'PLAK5a21e86c2faf452195d43c3ca3f318ee'

os.environ['PL_API_KEY'] = API_KEY
DOWNLOAD_DIR = os.getenv('TEST_DOWNLOAD_DIR', '.')
print(DOWNLOAD_DIR)
summer_2022 = ["2022-06-01T00:00:00.000Z", "2022-08-30T00:00:00.000Z",
               "2022-12-01T00:00:00.000Z", "2023-02-26T00:00:00.000Z"]

winter_2022 = ["2022-01-12T00:00:00.000Z", "2023-02-26T00:00:00.000Z",
               "2022-06-01T00:00:00.000Z", "2022-08-30T00:00:00.000Z"]

spring_2022 = ["2022-04-01T00:00:00.000Z", "2023-05-30T00:00:00.000Z",
               "2022-09-01T00:00:00.000Z", "2022-10-30T00:00:00.000Z"]

autumn_2022 = ["2022-09-01T00:00:00.000Z", "2022-10-30T00:00:00.000Z",
               "2022-04-01T00:00:00.000Z", "2023-05-30T00:00:00.000Z"]

summer_2023 = ["2023-06-01T00:00:00.000Z", "2023-08-30T00:00:00.000Z",
               "2023-12-01T00:00:00.000Z", "2023-02-26T00:00:00.000Z"]

winter_2023 = ["2023-01-12T00:00:00.000Z", "2023-02-26T00:00:00.000Z",
               "2023-06-01T00:00:00.000Z", "2023-08-30T00:00:00.000Z"]

spring_2023 = ["2023-04-01T00:00:00.000Z", "2023-05-30T00:00:00.000Z",
               "2023-09-01T00:00:00.000Z", "2023-10-30T00:00:00.000Z"]

autumn_2023 = ["2023-09-01T00:00:00.000Z", "2023-10-30T00:00:00.000Z",
               "2023-04-01T00:00:00.000Z", "2023-05-30T00:00:00.000Z"]

summer_2021 = ["2021-06-01T00:00:00.000Z", "2021-08-30T00:00:00.000Z",
               "2021-12-01T00:00:00.000Z", "2021-02-26T00:00:00.000Z"]

winter_2021 = ["2021-01-12T00:00:00.000Z", "2021-02-26T00:00:00.000Z",
               "2021-06-01T00:00:00.000Z", "2021-08-30T00:00:00.000Z"]

spring_2021 = ["2021-04-01T00:00:00.000Z", "2021-05-30T00:00:00.000Z",
               "2021-09-01T00:00:00.000Z", "2021-10-30T00:00:00.000Z"]

autumn_2021 = ["2021-09-01T00:00:00.000Z", "2021-10-30T00:00:00.000Z",
               "2021-04-01T00:00:00.000Z", "2021-05-30T00:00:00.000Z"]


PLANET_API_KEY = os.getenv('PL_API_KEY')
# Setup the API Key from the `PL_API_KEY` environment variable

BASE_URL = "https://api.planet.com/data/v1"

session = requests.Session()

# Authenticate session with user name and password, pass in an empty string for the password
session.auth = (PLANET_API_KEY, "")

res = '' #session.get(BASE_URL)


def select_image_ids(filename, position, gte, max_cloud_cover=0.1, visible_percent=100):
    """
    Select image ids for a position.

    :param filename:
    :param position:
    :param gte:
    :param max_cloud_cover:
    :param visible_percent:
    :return:
    """
    north_gte = gte[0]
    north_lte = gte[1]
    # print response body
    south_gte = gte[2]
    south_lte = gte[3]

    if position[0][0] < 0:
        gte = south_gte
        lte = south_lte
    else:
        gte = north_gte
        lte = north_lte

    geojson_geometry = {
        "type": "Polygon",
        "coordinates": [
            position
        ]
    }
    # get images that overlap with our AOI
    geometry_filter = {
        "type": "GeometryFilter",
        "field_name": "geometry",
        "config": geojson_geometry
    }

    # get images acquired within a date range
    date_range_filter = {
        "type": "DateRangeFilter",
        "field_name": "acquired",
        "config": {
            "gte": gte,
            "lte": lte
        }
    }

    # only get images which have <50% cloud coverage
    cloud_cover_filter = {
        "type": "RangeFilter",
        "field_name": "cloud_cover",
        "config": {
            "lte": max_cloud_cover
        }
    }

    # combine our geo, date, cloud filters
    combined_filter = {
        "type": "AndFilter",
        "config": [geometry_filter, cloud_cover_filter, date_range_filter],
    }

    item_type = "PSScene"

    # API request object
    search_request = {
        "item_types": [item_type],
        "filter": combined_filter
    }

    # fire off the POST request
    search_result = \
        requests.post(
            'https://api.planet.com/data/v1/quick-search',
            auth=HTTPBasicAuth(API_KEY, ''),
            json=search_request)

    geojson = search_result.json()

    geo_df = pd.DataFrame()  # Add in the things we're looking at
    ids = []
    azi = []
    cloud_cover = []
    visible_perc = []
    sun_azi = []
    # Also only want it if ortho_analytic_8b_sr is availabe in assets
    asset = 'ortho_analytic_8b_sr'
    for image in geojson['features']:
        if asset in image['assets']: # 'visible_percent'
            try:
                visible_perc.append(image['properties']['visible_percent'])
                cloud_cover.append(image['properties']['cloud_cover'])
                ids.append(image['id'])
                sun_azi.append(image['properties']['sun_azimuth'])
                azi.append(image['properties']['satellite_azimuth'])
            except:
                print(asset)
    geo_df['ids'] = ids
    geo_df['visible_percent'] = visible_perc
    geo_df['sun_azimuth'] = sun_azi
    geo_df['satellite_azimuth'] = azi
    geo_df['cloud_cover'] = cloud_cover

    geo_df = geo_df[geo_df['visible_percent'] >= visible_percent]
    # Swap visible percent to invisible so that we sort correctly
    geo_df['hidden_percent'] = 100 - geo_df['visible_percent'].values
    # Save this to CSV anyway so that the user can use it later

    # Get the mean azimuth i.e. optimise between sun and sat
    geo_df['mean_azi'] = (abs(abs(geo_df['sun_azimuth'].values) - 90) + abs(
        (abs(geo_df['satellite_azimuth'].values) - 90))) / 2
    geo_df = geo_df.sort_values(['cloud_cover', 'hidden_percent', 'mean_azi'])  # We want it to be as close to 90 degrees as possible
    geo_df.to_csv(filename, index=False)
    return geo_df['ids'].values[0]

def create_requests(poly, image_id, label):
    # The Orders API will be asked to mask, or clip, results to
    # this area of interest.
    iowa_aoi = {
        "type":
        "Polygon",
        "coordinates": [
            poly
        ]
    }

    # In practice, you will use a Data API search to find items, but
    # for this example take them as given.
    iowa_items = [image_id]

    iowa_order = planet.order_request.build_request(
        name=label,
        products=[
            planet.order_request.product(item_ids=iowa_items,
                                         product_bundle='analytic_8b_sr_udm2',
                                         item_type='PSScene')
        ],
        tools=[planet.order_request.clip_tool(aoi=iowa_aoi)])

    return iowa_order


async def create_and_download(client, order_detail, directory):
    """Make an order, wait for completion, download files as a single task."""
    with planet.reporting.StateBar(state='creating') as reporter:
        order = await client.create_order(order_detail)
        reporter.update(state='created', order_id=order['id'])
        await client.wait(order['id'], callback=reporter.update_state)

    await client.download_order(order['id'], directory, progress_bar=True)


async def download(data):
    async with planet.Session() as sess:
        client = sess.client('orders')
        requests = []
        for d in data:
            requests.append(create_requests(d[0], d[1], d[2]))

        print(await asyncio.gather(*[
            create_and_download(client, request, DOWNLOAD_DIR)
            for request in requests
        ], return_exceptions=True))
