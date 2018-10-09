#!/usr/bin/env python3

from dash import Dash
from dash.dependencies import Input, Output, State
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import numpy as np
import json
import base64
import urllib


def read_file(filename):
    """Reads the given file as loads its contents as json data"""
    with open(filename) as json_load:
        json_data = json.load(json_load)
        return json_data
    
    
def read_remote_file(file_url):
    """Reads a remote file and returns contents as json data.
    The given URL must point to a JSON file."""
    json_data = json.loads(urllib.request.urlopen(file_url).read())
    return json_data
    
    
def process_json_data(json_data):
    """Extracts out the number of groups and samples
    from the given json structure"""
    if not json_data:
        return [], []
    groups = json_data['groups']
    first_group = groups[0]
    sample_names = []
    for peak in first_group['peaks']:
        sample_names.append(peak['sampleName'])
        
    return list(range(1, len(groups) + 1)), sample_names


class GlobalInstance(object):
    _json_data = None
    filename = ""
    group_names, sample_names = [], []

    @property
    def json_data(self):
        return self._json_data

    @json_data.setter
    def json_data(self, data):
        self._json_data = data
        self.group_names, self.sample_names = process_json_data(data)


instance = GlobalInstance()


def alss(y, lam, p, niter=10):
    """Asymmetric Least Squares Smoothing is an iterative baseline
    estimation algorithm.
    
    Ref:
    Baseline Correction with Asymmetric Least Squares Smoothing,
    P. Eilers, H. Boelens, 2005
    
    :param y: Signal data as numpy array
    :param lam: Lambda integer value (range 10^2 to 10^8 suits MS data)
    :param p: Assymetry factor (range 0.001 to 0.1 suitable for positive peaks)
    :param niter: Number of iterations to run for finding suitable baseline
    
    :returns: A numpy array (same size as `y`) as the estimated baseline
    """
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


def trim_and_find_bounds(arr):
    """Some datasets have leading and trailing zeros in their intensity vector.
    These zeros can sometimes have an effect on the estimated baseline (tapering
    down on the edges of the signal). Processing the intensity array with this
    function can provide better (or worse) results."""
    non_zeros = np.where(arr != 0)
    if (non_zeros[0].any()):
        first = non_zeros[0][0]
        last = non_zeros[0][-1]
        arr_trim = np.trim_zeros(arr)
        return arr_trim, first, last + 1
    
    return arr, 0, arr.size


def calculate_baseline(json_data, group_id, sample, lam, p):
    """Given the JSON data, group, sample and parameters, calculates and
    returns the baseline.

    :param json_data: JSON data from a file that we read the EICs from
    :param group_id: String ID for the group to display
    :param sample: String sample name for the sample to display. If sample is
    `None` then the first group is operated on.
    :param lam: 10 raised to this lambda value will be passed as ALSS parameter
    :param p: Asymmetry factor that will be passed as ALSS parameter

    :returns: A tuple containing rt, intensity and calculated baseline arrays
    """
    if not json_data:
        print("Error: JSON data not loaded. Please check file path/URL.")
        return
    
    groups = json_data['groups']
    for group in groups:
        if group['groupId'] != group_id:
            continue

        for peak in group['peaks']:
            if sample and peak['sampleName'] != sample:
                continue

            rt = np.array(peak['eic']['rt'])
            ints = np.array(peak['eic']['intensity'])

            # apply baseline correction
            # correction needed for value of `p` since it is an integer (0 < p < 100) from slider value
            # correction needed for value of `lam` since it is an integer (0 < p < 10) from slider value
            z = alss(ints, 10 ** lam, p/1000)

            # remove the negative values in the obtained baseline
            np.clip(z, 0, np.inf, out=z)

            return rt, ints, z

    print("Error: Given group or sample not found.")


example_json_file = 'example_output.json'
example_data = read_file(example_json_file)
example_groups, example_samples = process_json_data(example_data)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions']=True
app.layout = html.Div([

    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label="Example EICs", value='tab-1'),
        dcc.Tab(label="Custom EICs", value='tab-2'),
    ],
    style={'marginBottom': '6px'}),

    html.Div(id='tabs-content'),

], style={'width': '92%', 'marginLeft': '4%', 'display': 'inline-block'})

server = app.server

spacer_top_small = html.Div([], style={'marginTop': 6,})
spacer_top_large = html.Div([], style={'marginTop': 24,})
spacer_bottom_small = html.Div([], style={'marginBottom': 6})
spacer_bottom_large = html.Div([], style={'marginBottom': 24})


def eic_content(tab):
    return html.Div([

        spacer_top_small,

        dcc.Graph(
            id='eic-' + tab,
            style={'width': '50%', 'height': '360px', 'display': 'inline-block'}
        ),

        dcc.Graph(
            id='corrected-eic-' + tab,
            style={'width': '50%', 'height': '360px', 'display': 'inline-block', 'float': 'right'}
        ),

        spacer_top_small,

        html.Div([
            dcc.Markdown("**Smoothness** (λ): "),
            dcc.Slider(
                id='lambda-slider-' + tab,
                min=0,
                max=9,
                value=4,
                marks={lam: lam for lam in range(0, 10)},
            )
        ]),

        html.Div(style={'marginBottom': '24px', 'marginTop': '24px'}),

        html.Div([
            dcc.Markdown("**Assymetry** (p): "),
            dcc.Slider(
                id='p-slider-' + tab,
                min=0,
                max=100,
                value=30,
                step=5,
                marks={p: p/1000 for p in range(0, 101, 10)},
            ),
        ]),

        spacer_bottom_small

    ])


example_eics_content = html.Div([

    spacer_top_small,

    dcc.Dropdown(
        id = 'example-dropdown',
        options = [{'label': "Example EIC " + str(group), 'value': group} for group in example_groups],
        value = example_groups[0]
    ),

    spacer_bottom_small
])

loaded_eics_content = html.Div([

    html.Div([

        html.Div([

            dcc.Upload(
                id='json-loader',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select JSON File'),
                    ' exported from El-MAVEN'
                ]),
                style={
                    'width': '100%',
                    'height': '76px',
                    'lineHeight': '76px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'marginBottom': '6px'
                },
            ),

        ])
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div(
        id='post-upload-dropdowns', 
        style={'width': '48%', 'float': 'right', 'display': 'inline-block'}
    )

], style={'marginBottom': '18px'})


@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            example_eics_content,
            eic_content(tab)
        ])
    elif tab == 'tab-2':
        return html.Div([
            loaded_eics_content,
            eic_content(tab)
        ])


def parse_contents(contents, filename, date):
    _, content_string = contents.split(',')

    utf8_content = base64.b64decode(content_string)
    try:
        if 'json' in filename:
            # Assume that the user uploaded a JSON file
            instance.json_data = json.loads(utf8_content.decode('utf-8'))
            instance.filename = filename
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([

        dcc.Dropdown(
            id='group-dropdown',
            options = [{'label': group, 'value': group} for group in instance.group_names],
            value = instance.group_names[0]
        ),

        spacer_bottom_small,

        dcc.Dropdown(
            id='sample-dropdown',
            options = [{'label': sample, 'value': sample} for sample in instance.sample_names],
            value = instance.sample_names[0]
        )

    ])


@app.callback(Output('post-upload-dropdowns', 'children'),
              [Input('json-loader', 'contents')],
              [State('json-loader', 'filename'),
               State('json-loader', 'last_modified')])
def update_output(contents, filename, last_modified):
    if contents is not None:
        return [parse_contents(contents, filename, last_modified)]


@app.callback(Output('eic-tab-1', 'figure'), [Input('example-dropdown', 'value'),
                                              Input('lambda-slider-tab-1', 'value'),
                                              Input('p-slider-tab-1', 'value')])
def update_example_graph(group_id, lam, p):
    """This function takes in relevant parameters and plots the EIC for
    given group from example EICs.
    """
    rt, ints, z = calculate_baseline(example_data, group_id, None, lam, p)
    return {
        'data': [
            {
                'x': rt,
                'y': ints,
                'text': "Signal",
                'name': "Signal",
            },
            {
                'x': rt,
                'y': z,
                'text': "Baseline",
                'name': "ALSS Baseline",
            }
        ],
        'layout': go.Layout(
            title="Original EIC",
            xaxis={'title': "Retention Time"},
            yaxis={'title': "Intensity"},
            margin={'b': 40, 't': 40, 'r': 10},
            legend={'x': 0, 'y': 1},
        )
    }


@app.callback(Output('corrected-eic-tab-1', 'figure'), [Input('example-dropdown', 'value'),
                                                        Input('lambda-slider-tab-1', 'value'),
                                                        Input('p-slider-tab-1', 'value')])
def update_example_corrected_graph(group_id, lam, p):
    """Redraws the corrected EIC. See `update_example_graph` for more information."""
    rt, ints, z = calculate_baseline(example_data, group_id, None, lam, p)

    # subtracted baseline from original signal
    corrected_ints = ints - z

    return {
        'data': [
            {
                'x': rt,
                'y': corrected_ints,
                'text': "Corrected Signal",
                'name': "Corrected Signal",
            }
        ],
        'layout': go.Layout(
            title="Corrected EIC",
            xaxis={'title': "Retention Time"},
            yaxis={'title': "Intensity"},
            margin={'b': 40, 't': 40, 'r': 10},
            legend={'x': 0, 'y': 1},
        )
    }


@app.callback(Output('eic-tab-2', 'figure'), [Input('group-dropdown', 'value'),
                                              Input('sample-dropdown', 'value'),
                                              Input('lambda-slider-tab-2', 'value'),
                                              Input('p-slider-tab-2', 'value')])
def update_graph(group_id, sample, lam, p):
    """This function takes in relevant parameters and plots the EIC for
    given group and sample name.
    """
    rt, ints, z = calculate_baseline(instance.json_data, group_id, sample, lam, p)
    return {
        'data': [
            {
                'x': rt,
                'y': ints,
                'text': "Signal",
                'name': "Signal",
            },
            {
                'x': rt,
                'y': z,
                'text': "Baseline",
                'name': "ALSS Baseline",
            }
        ],
        'layout': go.Layout(
            title="Original EIC",
            xaxis={'title': "Retention Time"},
            yaxis={'title': "Intensity"},
            margin={'b': 40, 't': 40, 'r': 10},
            legend={'x': 0, 'y': 1},
        )
    }


@app.callback(Output('corrected-eic-tab-2', 'figure'), [Input('group-dropdown', 'value'),
                                                        Input('sample-dropdown', 'value'),
                                                        Input('lambda-slider-tab-2', 'value'),
                                                        Input('p-slider-tab-2', 'value')])
def update_corrected_graph(group_id, sample, lam, p):
    """Redraws the corrected EIC. See `update_graph` for more information."""
    rt, ints, z = calculate_baseline(instance.json_data, group_id, sample, lam, p)

    # subtracted baseline from original signal
    corrected_ints = ints - z

    return {
        'data': [
            {
                'x': rt,
                'y': corrected_ints,
                'text': "Corrected Signal",
                'name': "Corrected Signal",
            }
        ],
        'layout': go.Layout(
            title="Corrected EIC",
            xaxis={'title': "Retention Time"},
            yaxis={'title': "Intensity"},
            margin={'b': 40, 't': 40, 'r': 10},
            legend={'x': 0, 'y': 1},
        )
    }


if __name__ == '__main__':
    app.run_server(debug=True)
