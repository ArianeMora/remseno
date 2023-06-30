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

import asyncio
import os
# import os module to access enviornmental modules

import requests

from planet import Session

os.environ['PL_API_KEY'] = '86d667757ba14bc38dd4555d8ab948d5'
import asyncio
import os
import planet

DOWNLOAD_DIR = os.getenv('TEST_DOWNLOAD_DIR', '.')


def create_requests(poly, image_id):
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
        name='iowa_order',
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
            requests.append(create_requests(d[0], d[1]))

        await asyncio.gather(*[
            create_and_download(client, request, DOWNLOAD_DIR)
            for request in requests
        ], return_exceptions=True)