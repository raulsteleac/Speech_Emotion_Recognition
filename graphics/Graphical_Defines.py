import os

COMPLETED_STYLE_ANGRY = """
QProgressBar {
    border: 1px solid #76797C;
    border-radius: 5px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: red;
}
"""

COMPLETED_STYLE_HAPPY = """
QProgressBar {
    border: 1px solid #76797C;
    border-radius: 5px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: yellow;
}
"""

COMPLETED_STYLE_SAD= """
QProgressBar {
    border: 1px solid #76797C;
    border-radius: 5px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: gray;
}
"""
SLYDER_ENABLED = """
QSlider::groove:horizontal {
    border: 1px solid #565a5e;
    height: 4px;
    background: #565a5e;
    margin: 0px;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #D1DBCB;
    border: 1px solid #999999;
    width: 10px;
    height: 10px;
    margin: -5px 0;
}

QSlider::add-page:qlineargradient {
    background: #595858;
    border-top-right-radius: 5px;
    border-bottom-right-radius: 5px;
    border-top-left-radius: 0px;
    border-bottom-left-radius: 0px;
}

QSlider::sub-page::qlineargradient:horizontal {
    background:  #D1DBCB;
    border-top-right-radius: 0px;
    border-bottom-right-radius: 0px;
    border-top-left-radius: 5px;
    border-bottom-left-radius: 5px;
}"""
SLYDER_DISABLED = """

QSlider::groove:horizontal {
    border: 1px solid #565a5e;
    height: 4px;
    background: #595858;
    margin: 0px;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #595858;
    border: 1px solid #999999;
    width: 10px;
    height: 10px;
    margin: -5px 0;
}

QSlider::add-page:qlineargradient {
    background: #595858;
    border-top-right-radius: 5px;
    border-bottom-right-radius: 5px;
    border-top-left-radius: 0px;
    border-bottom-left-radius: 0px;
}

QSlider::sub-page::qlineargradient:horizontal {
    background:  #595858;
    border-top-right-radius: 0px;
    border-bottom-right-radius: 0px;
    border-top-left-radius: 5px;
    border-bottom-left-radius: 5px;
}"""

map_config = {
    "EMO-DB": 1,
    "SAVEE": 2,
    "RAVDESS": 3,
    "ENTERFACE": 4,
    "EMOVO": 5,
    "MAV": 6, 
    "MELD": 7,
    "JL": 8,
    "INRP": 9,
    "MULTIPLE": 10,
}