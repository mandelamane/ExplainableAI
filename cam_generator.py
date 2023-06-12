from dataclasses import dataclass, field
from typing import Tuple

import cv2
import numpy as np
from keras.models import Model


@dataclass
class ScoreCamSaliency:
    """
    ScoreCamSaliency represents a method to generate class activation maps
    using a trained CNN model. It is designed to interpret the model's
    predictions
    for seismic event types, such as Noise, Tectonic Tremor, and Earthquake,
    by visualizing the important areas in the input spectrograms.

    Attributes
    ----------
    model : Model
        The trained CNN model.
    target_index : int
        The index of the seismic event type that the class activation map will
        represent. The indices correspond to the following event types:
            - '0': Noise
            - '1': Tectonic Tremor
            - '2': Earthquake
    layer_name : str
        The name of the convolutional layer in the CNN model that will be used
        for generating the class activation map.
    extracted_model : Model
        A new model, derived from the original model, that includes only the
        layers up to and including the specified convolutional layer.
        This model is used to generate class activation maps.
    input_shape : Tuple[int, int, int]
        The shape of the input data expected by the CNN model. It includes the
        height, width, and the number of channels of the input data.
    """

    model: Model
    target_index: int
    layer_name: str

    extracted_model: Model = field(init=False)
    input_shape: Tuple[int, int, int] = field(init=False)

    def __post_init__(self):
        """
        Initializes the ScoreCamSaliency instance and extracts the
        convolutional layer from the model.
        """
        self._extract_model()
        self.input_shape = self.model.layers[0].output_shape[0][1:]

    def _extract_model(self):
        """
        Extracts the specified convolutional layer from the trained model.
        This extracted model will be used for generating class activation maps.
        """
        self.extracted_model = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(self.layer_name).output,
        )

    def __call__(self, spectrograms: np.ndarray) -> np.ndarray:
        """
        Computes the class activation maps (saliency maps) for the provided
        spectrograms using the Score-CAM method.

        Parameters
        ----------
        spectrograms : np.ndarray
            A 4D array containing spectrogram data for each station and each
            component.

        Returns
        -------
        np.ndarray
            A 3D array representing the saliency maps (class activation maps)
            for each station.
        """
        conv_output_array = self.extracted_model.predict(
            spectrograms, verbose=0
        )
        y_prob = self.model.predict(spectrograms, verbose=0)

        num_features = conv_output_array.shape[-1]
        num_stations = spectrograms.shape[0]
        feature_height, feature_width = self.input_shape[:2]

        saliency_maps_resized = np.empty(
            (num_stations, num_features, feature_height, feature_width)
        )

        for i in range(num_stations):
            for num_feature in range(num_features):
                saliency_maps_resized[i, num_feature] = cv2.resize(
                    conv_output_array[i, :, :, num_feature],
                    self.input_shape[:2][::-1],
                    interpolation=cv2.INTER_LINEAR,
                )

        norms = np.max(saliency_maps_resized, axis=(2, 3)) - np.min(
            saliency_maps_resized, axis=(2, 3)
        )

        non_zero_norm_mask = norms != 0

        norm_saliency_maps = (
            saliency_maps_resized
            - np.min(saliency_maps_resized, axis=(2, 3))[
                :, :, np.newaxis, np.newaxis
            ]
        )

        norm_saliency_maps[non_zero_norm_mask] /= norms[non_zero_norm_mask][
            :, np.newaxis, np.newaxis
        ]

        masked_spectrograms = (
            spectrograms[:, np.newaxis, :, :, :]
            * norm_saliency_maps[:, :, :, :, np.newaxis]
        )

        # Flattening masked_spectrograms to (num_stations*num_features,
        # feature_height, feature_width, num_channels)
        # where num_channels is the last dimension of spectrograms
        num_channels = spectrograms.shape[-1]
        masked_spectrograms_flatten = masked_spectrograms.reshape(
            -1, feature_height, feature_width, num_channels
        )

        scores = (self.model.predict(masked_spectrograms_flatten, verbose=1))[
            :, self.target_index
        ]

        scores_reshaped = scores.reshape(num_stations, num_features)

        all_station_score_map = np.einsum(
            "ijkl,ij->ikl", saliency_maps_resized, scores_reshaped
        )

        all_station_score_map = np.maximum(0, all_station_score_map)
        all_station_score_map /= np.max(all_station_score_map, axis=(1, 2))[
            :, np.newaxis, np.newaxis
        ]
        return all_station_score_map, y_prob

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the softmax of a given numpy array. The softmax function is
        used to transform the array into a probability distribution.

        Parameters
        ----------
        x : np.ndarray
            The input numpy array for which to compute the softmax.

        Returns
        -------
        np.ndarray
            The computed softmax of the input numpy array.
        """
        f = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return f
