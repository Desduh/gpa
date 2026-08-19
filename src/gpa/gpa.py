import numpy as np
from math import atan2, sqrt, pi
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from numpy.typing import NDArray

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 100,      
    "savefig.dpi": 300,     
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
})


class GPA:
    def __init__(self, matrix: NDArray[np.number]):
        """
        Gradient Pattern Analysis (GPA) implementation.

        This class implements the Gradient Pattern Analysis (GPA) method for
        two-dimensional images. Given an input image, it computes the gradient
        field, removes radially symmetric gradient vectors with respect to a
        reference center, and evaluates the four GPA moments (G1, G2, G3, and G4).

        Reference
        ---------
        Paper:

        Example
        -------
        >>> gpa = GPA(image)
        >>> gpa.setPosition(cx, cy)
        >>> results = gpa.evaluate()

        The analysis center is defined with ``setPosition(cx, cy)``, where
        ``(cx, cy)`` corresponds to the point from which radial symmetry is
        evaluated. By default, the geometric center of the image is used.

        Parameters
        ----------
        matrix : array-like
            Two-dimensional input image :math:`I(x, y)`. The input can be
            provided as a NumPy array or as any array-like object that can
            be converted to a NumPy array, such as a nested Python list.

            The matrix must contain real numerical values and have at least
            3 rows and 3 columns.
        """

        # Convert the input to a NumPy array.
        #
        # This allows the class to accept both NumPy arrays and
        # manually defined matrices, such as nested Python lists.
        try:
            matrix = np.asarray(matrix)
        except Exception as exc:
            raise TypeError(
                "Input matrix must be convertible to a NumPy array."
            ) from exc

        # The matrix must contain real numerical values.
        if not np.issubdtype(matrix.dtype, np.number) or np.iscomplexobj(matrix):
            raise TypeError(
                "Input matrix must contain only real numerical values."
            )

        # The input must be two-dimensional.
        if matrix.ndim != 2:
            raise ValueError(
                "Input matrix must be two-dimensional."
            )

        # GPA requires a minimum matrix size of 3 × 3.
        if matrix.shape[0] < 3 or matrix.shape[1] < 3:
            raise ValueError(
                "Input matrix must have at least 3 rows and 3 columns."
            )

        # Convert the input matrix to float32.
        self.matrix = matrix.astype(np.float32, copy=False)

        # Image dimensions
        self.rows, self.cols = self.matrix.shape

        # Coordinates of the image center
        self.cx = (self.cols - 1) / 2
        self.cy = (self.rows - 1) / 2

        # Gradient field components:
        #
        # ∇I = (Gx, Gy)
        #
        self.gradient_dx = None
        self.gradient_dy = None

        # Gradient field after removing
        # radially symmetric vectors
        self.gradient_asymmetric_dx = None
        self.gradient_asymmetric_dy = None

        # Gradient properties:
        #
        # phases -> orientation:
        # θ = atan2(Gy, Gx)
        #
        # mods -> magnitude:
        # |∇I| = sqrt(Gx² + Gy²)
        #
        self.phases = None
        self.mods = None

        # Coordinates of the removed points and
        # the remaining asymmetric points
        self.removedP = np.empty((0, 2), dtype=np.int32)
        self.nremovedP = np.empty((0, 2), dtype=np.int32)

        # Total number of gradient vectors and
        # remaining asymmetric vectors
        self.totalVet = self.rows * self.cols
        self.totalAssimetric = self.rows * self.cols

        # Measure of gradient orientation diversity
        self.phaseDiversity = 0.0

        # Maximum gradient magnitude found in the image
        self.maxGrad = 0.0



    def setPosition(self, cx: float, cy: float):
        """
        Set the reference center used in the GPA analysis.

        If this method is not called, the geometric center of the input image
        is used by default.

        Parameters
        ----------
        cx : float
            X-coordinate of the reference center.
        cy : float
            Y-coordinate of the reference center.
        """
        self.cx = float(cx)
        self.cy = float(cy)



    def evaluate(
        self,
        magnitude_threshold=0.01,
        magnitude_tolerance=None,
        angle_tolerance=None,
        radial_distance_tolerance=0.5,
        symmetric_position_tolerance=0.01,
        opposite_vector_tolerance=0.3,
        mask=None,
        moments=["G1", "G2", "G3", "G4"],
    ):
        """
        Perform Gradient Pattern Analysis (GPA) and compute the selected
        gradient moments.

        The analysis consists of the following steps:

        1. Compute the image gradient field:

            ∇I(x, y) = (Gx, Gy)

        2. Group gradient vectors according to the radial distance of their
        positions from the analysis center.

        3. Remove gradient vectors whose magnitudes are below the specified
        relative magnitude threshold.

        4. Identify and remove radially symmetric gradient-vector pairs
        according to either:

        - a vector-sum criterion, controlled by ``opposite_vector_tolerance``;
            or
        - separate magnitude and angular criteria, controlled by
            ``magnitude_tolerance`` and ``angle_tolerance``.

        5. Compute the selected GPA moments:

            G1, G2, G3, and G4.

        Parameters
        ----------
        magnitude_threshold : float, optional
            Relative threshold for the gradient magnitude (fraction of the
            maximum gradient magnitude, dimensionless). Vectors satisfying

                |∇I| / max(|∇I|) <= magnitude_threshold

            are removed. For example, ``0.01`` corresponds to 1%.

        magnitude_tolerance : float or None, optional
            Relative tolerance for comparing the magnitudes of two gradient
            vectors (fraction of the maximum gradient magnitude, dimensionless).
            It is used only when ``opposite_vector_tolerance`` is ``None``.
            For example, ``0.05`` corresponds to 5%.

        angle_tolerance : float or None, optional
            Angular tolerance for comparing gradient-vector orientations
            (degrees). It is used only when ``opposite_vector_tolerance`` is
            ``None``. The value is internally converted to radians.
            For example, ``5.0`` corresponds to a tolerance of ±5 degrees
            around the opposite orientation.

        radial_distance_tolerance : float, optional
            Tolerance used when grouping pixels according to their radial
            distance from the analysis center (pixels). Pixels whose radial
            distances differ from a given radial group by at most this value
            are considered part of the same group.

        symmetric_position_tolerance : float, optional
            Relative spatial tolerance used to determine whether two pixels
            are approximately opposite with respect to the analysis center
            (percentage of the image dimensions, dimensionless).

            The tolerance is converted independently for each image dimension:

                tolerance_x = symmetric_position_tolerance / 100 * image_width
                tolerance_y = symmetric_position_tolerance / 100 * image_height

            For example, ``0.01`` corresponds to 0.01% of the image width
            and height.

        opposite_vector_tolerance : float or None, optional
            Relative residual tolerance for the vector-sum criterion
            (dimensionless). Two gradient vectors are considered approximately
            opposite when the magnitude of their vector sum, normalized by the
            sum of their individual magnitudes, satisfies

                |G₁ + G₂| / (|G₁| + |G₂|) <= opposite_vector_tolerance.

            A value of ``0`` requires perfectly opposite vectors, while larger
            values allow progressively larger deviations. For example,
            ``0.3`` allows a normalized residual of up to 30%.

            If this parameter is not ``None``, ``magnitude_tolerance`` and
            ``angle_tolerance`` are ignored.

        mask : numpy.ndarray or None, optional
            Binary mask defining the valid image region (dimensionless).
            Pixels with mask value 0 are excluded from the analysis. If
            ``None``, all pixels are considered valid.

        moments : str, list, tuple, or None, optional
            GPA moments to compute (dimensionless identifiers). Valid options
            are ``"G1"``, ``"G2"``, ``"G3"``, and ``"G4"``. A single moment
            may be provided as a string. If ``None``, all four moments are
            computed.

        Returns
        -------
        dict
            Dictionary containing the requested GPA moments.
        """

        if opposite_vector_tolerance is not None:
            opposite_vector_tolerance = np.float32(opposite_vector_tolerance)
            magnitude_tolerance = None
            angle_tolerance = None
        else: 
            if magnitude_tolerance is None:
                magnitude_tolerance = 0.05
                print("magnitude_tolerance not provided. Using 0.05 as the default value.")

            if angle_tolerance is None:
                angle_tolerance = 0.05
                print("angle_tolerance not provided. Using 0.05 as the default value.")

        if angle_tolerance is not None:
            angle_tolerance = np.deg2rad(angle_tolerance)

        self.mask = mask
        if mask is None:
            self.mask = np.ones_like(self.matrix, dtype=np.float32)

        # Compute the image gradient field, as well as the
        # corresponding magnitudes and orientations.
        self._setGradients()


        # Update the image dimensions
        self.cols = len(self.matrix[0])
        self.rows = len(self.matrix)


        # Compute the radial distance map:
        #
        # r = sqrt((x - cx)² + (y - cy)²)
        #
        # Each pixel is assigned its integer radial distance
        # from the analysis center.
        radial_distance_map = np.array([
            [
                int(np.sqrt((x - self.cx)**2 + (y - self.cy)**2))
                for x in range(self.cols)
            ]
            for y in range(self.rows)
        ], dtype=np.int32)


        # Retrieve the unique radial distances present in the image
        unique_radii = np.unique(radial_distance_map).astype(np.int32)


        # Remove radially symmetric gradient vectors,
        # producing the asymmetric gradient field.
        self._update_asymmetric_mat(
            unique_radii,
            radial_distance_map,
            magnitude_threshold,
            magnitude_tolerance,
            angle_tolerance,
            radial_distance_tolerance,
            symmetric_position_tolerance,
            opposite_vector_tolerance
        )


        # Allow a single GPA moment to be specified as a string
        if isinstance(moments, str):
            moments = [moments]

        elif isinstance(moments, tuple):
            moments = list(moments)


        # Map each GPA moment name to its corresponding function
        available_moments = {
            "G1": self._G1,
            "G2": self._G2,
            "G3": self._G3,
            "G4": self._G4
        }


        results = {}

        # Compute the requested GPA moments
        for moment in moments:

            if moment not in available_moments:
                raise ValueError(
                    f"Invalid GPA moment '{moment}'. "
                    f"Available options are: {list(available_moments.keys())}"
                )

            results[moment] = available_moments[moment]()

        self.radial_distance_map = radial_distance_map

        return results


        
    def _setGradients(self):
        """
        Compute the image gradient field and its associated properties.

        For an image I(x, y), the gradient is defined as:

            ∇I(x, y) = (Gx, Gy)

        where:

            Gx = ∂I/∂x

        is the intensity variation along the horizontal direction, and

            Gy = ∂I/∂y

        is the intensity variation along the vertical direction.

        The gradient magnitude is computed as:

            |∇I| = sqrt(Gx² + Gy²)

        which measures the strength of the local intensity variation.

        The gradient orientation is given by:

            θ = atan2(Gy, Gx)

        representing the angle of the gradient vector with respect to the
        x-axis.

        These quantities are subsequently used by the GPA algorithm to
        identify asymmetric structures through comparisons of gradient
        vector magnitudes and orientations.
        """

        # Compute the horizontal (gx) and vertical (gy)
        # gradients of the input image.
        #
        # gx = ∂I/∂x
        # gy = ∂I/∂y
        gy, gx = self.gradient(self.matrix)

        # Store the original gradient components.
        # These arrays remain unchanged throughout the analysis.
        self.gradient_dx = gx.copy()
        self.gradient_dy = gy.copy()

        # Create copies of the gradient components that will
        # be modified during the symmetry-removal procedure.
        self.gradient_asymmetric_dx = gx.copy()
        self.gradient_asymmetric_dy = gy.copy()

        # Compute the gradient magnitude:
        #
        # |∇I| = sqrt(Gx² + Gy²)
        self.mods = np.sqrt(gx**2 + gy**2)

        # Store the maximum gradient magnitude.
        # This value is used later for normalization:
        #
        # |∇I| / max(|∇I|)
        self.maxGrad = self.mods.max()

        # Compute the gradient orientation:
        #
        # θ = atan2(Gy, Gx)
        #
        # The result is initially in the interval [-π, π].
        angle = np.arctan2(gy, gx)

        # Convert negative angles to the interval [0, 2π].
        self.phases = np.where(
            angle >= 0,
            angle,
            angle + 2 * np.pi
        ).astype(np.float32)



    def gradient(self, matrix):
        """
        Compute the spatial gradient of a two-dimensional image using
        finite differences.

        For an image I(x, y), the gradient is defined as:

            ∇I(x, y) = (Gx, Gy)

        where:

            Gx = ∂I/∂x
            Gy = ∂I/∂y

        Central differences are used for interior pixels:

            Gx = [I(x+1, y) - I(x-1, y)] / 2
            Gy = [I(x, y+1) - I(x, y-1)] / 2

        Forward and backward differences are used at the image boundaries.

        Parameters
        ----------
        matrix : numpy.ndarray
            Two-dimensional input image.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple ``(dy, dx)`` containing the vertical and horizontal
            gradient components, respectively.
        """

        # Image dimensions
        h, w = matrix.shape

        # Allocate the gradient component arrays
        dx = np.zeros((h, w), dtype=np.float32)
        dy = np.zeros((h, w), dtype=np.float32)

        # Compute the gradient at each pixel
        for j in range(h):
            for i in range(w):

                # Vertical gradient (∂I/∂y)
                #
                # Use central differences for interior pixels.
                if 0 < j < h - 1:
                    dy[j, i] = (matrix[j + 1, i] - matrix[j - 1, i]) / 2.0

                # Use forward differences along the top boundary.
                elif j < h - 1:
                    dy[j, i] = matrix[j + 1, i] - matrix[j, i]

                # Use backward differences along the bottom boundary.
                elif j > 0:
                    dy[j, i] = matrix[j, i] - matrix[j - 1, i]

                # Horizontal gradient (∂I/∂x)
                #
                # Use central differences for interior pixels.
                if 0 < i < w - 1:
                    dx[j, i] = (matrix[j, i + 1] - matrix[j, i - 1]) / 2.0

                # Use forward differences along the left boundary.
                elif i < w - 1:
                    dx[j, i] = matrix[j, i + 1] - matrix[j, i]

                # Use backward differences along the right boundary.
                elif i > 0:
                    dx[j, i] = matrix[j, i] - matrix[j, i - 1]

        return dy, dx



    def _update_asymmetric_mat(
        self,
        unique_radii,
        radial_distance_map,
        magnitude_threshold,
        magnitude_tolerance,
        angle_tolerance,
        radial_distance_tolerance,
        symmetric_position_tolerance,
        opposite_vector_tolerance
    ):
        """
        Remove radially symmetric contributions from the gradient field.

        Starting from the gradient field

            ∇I(x, y) = (Gx, Gy),

        the gradient magnitude and orientation are defined as

            |∇I| = sqrt(Gx² + Gy²)

            θ = atan2(Gy, Gx).

        The algorithm groups pixels according to their radial distance from
        the analysis center and compares gradient vectors within each group.

        Gradient vectors whose magnitude is below the specified relative
        threshold are first removed. The remaining vectors are then tested
        for radial symmetry.

        Two gradient vectors can be classified as a symmetric pair using
        one of two criteria.

        When ``opposite_vector_tolerance`` is provided, the vector-sum
        criterion is used:

            |G₁ + G₂|
            --------- <= opposite_vector_tolerance
            |G₁| + |G₂|

        This criterion directly measures how close the two vectors are to
        being equal in magnitude and opposite in direction.

        Otherwise, symmetry is determined using three independent conditions:

        1. Similar gradient magnitudes:

            ||∇I₁| - |∇I₂|| <= magnitude_tolerance * max(|∇I|)

        2. Approximately opposite orientations:

            |Δθ - π| <= angle_tolerance

        3. Approximately opposite spatial positions with respect to the
        analysis center:

            |x₁ + x₂ - 2cx| <= tolerance_x

            |y₁ + y₂ - 2cy| <= tolerance_y

        where ``(cx, cy)`` is the analysis center.

        When all required symmetry conditions are satisfied, both vectors
        are removed by setting

            Gx = 0
            Gy = 0

        resulting in the asymmetric gradient field

            ∇I_asym = (Gx_asym, Gy_asym).

        Parameters
        ----------
        unique_radii : numpy.ndarray
            Array containing the distinct radial distances present in the
            image (pixels).

        radial_distance_map : numpy.ndarray
            Two-dimensional array containing the radial distance of each
            pixel from the analysis center (pixels).

        magnitude_threshold : float
            Relative threshold for the gradient magnitude (fraction of the
            maximum gradient magnitude, dimensionless). Vectors satisfying

                |∇I| / max(|∇I|) <= magnitude_threshold

            are removed. For example, ``0.01`` corresponds to 1%.

        magnitude_tolerance : float or None
            Relative tolerance for comparing gradient magnitudes
            (fraction of the maximum gradient magnitude, dimensionless).
            This criterion is used only when ``opposite_vector_tolerance``
            is ``None``. For example, ``0.05`` corresponds to 5%.

        angle_tolerance : float or None
            Angular tolerance for comparing gradient orientations (radians
            internally). The user-facing value is specified in degrees and
            converted to radians before this function is called.

        radial_distance_tolerance : float
            Tolerance used when assigning pixels to the same radial group
            (pixels).

        symmetric_position_tolerance : float
            Relative tolerance for comparing the spatial positions of two
            pixels with respect to the analysis center (percentage of the
            corresponding image dimension, dimensionless).

            The actual tolerances used by the algorithm are

                tolerance_x = symmetric_position_tolerance / 100 * cols

                tolerance_y = symmetric_position_tolerance / 100 * rows

            where ``cols`` and ``rows`` are the image width and height,
            respectively.

        opposite_vector_tolerance : float or None
            Relative tolerance for the vector-sum symmetry criterion
            (dimensionless). The normalized residual is defined as

                |G₁ + G₂|
                ---------
                |G₁| + |G₂|.

            The vectors are considered opposite when this quantity is less
            than or equal to ``opposite_vector_tolerance``. For example,
            ``0.3`` corresponds to a maximum normalized residual of 30%.

            If provided, ``magnitude_tolerance`` and ``angle_tolerance`` are
            not used.
        """

        # Convert the arrays to the same data types used by the
        # original Cython implementation.
        mask = np.asarray(self.mask, dtype=np.float32)
        unique_radii = np.asarray(unique_radii, dtype=np.int32)
        radial_distance_map = np.asarray(radial_distance_map, dtype=np.int32)

        # Convert the tolerance values to float32.
        angle_tolerance = np.float32(angle_tolerance)
        radial_distance_tolerance = np.float32(radial_distance_tolerance)
        symmetric_position_tolerance = np.float32(symmetric_position_tolerance)

        removedP = []

        # Process each radial distance.
        # Gradient vectors belonging to the same radial group
        # may form symmetric pairs.
        for radius in unique_radii:

            x2 = []
            y2 = []

            # Collect all pixels belonging to the current
            # radial distance.
            for py in range(self.rows):
                for px in range(self.cols):

                    if abs(radial_distance_map[py, px] - radius) <= abs(radial_distance_tolerance):
                        x2.append(px)
                        y2.append(py)

            # Pixel coordinates within the current radial group.
            x = np.array(x2, dtype=np.int32)
            y = np.array(y2, dtype=np.int32)

            lx = len(x)

            # Compare every pair of gradient vectors in the
            # current radial group.
            # cont = 0
            for i in range(lx):

                px = x[i]
                py = y[i]

                if mask[py, px] == 0:
                    continue


                if (
                    self.gradient_asymmetric_dx[py, px] == 0.0
                    and self.gradient_asymmetric_dy[py, px] == 0.0
                ):
                    continue

                # Remove gradient vectors whose magnitude is
                # below the specified threshold.
                if (self.mods[py, px] / self.maxGrad) <= magnitude_threshold:

                    self.gradient_asymmetric_dx[py, px] = np.float32(0.0)
                    self.gradient_asymmetric_dy[py, px] = np.float32(0.0)

                    continue

                # Compare the current vector with the remaining
                # vectors in the same radial group.
                for j in range(i + 1, lx):

                    px2 = x[j]
                    py2 = y[j]

                    # if mask[py2, px2] == 0:
                    #     continue

                    if mask[py2, px2] == 0:
                        continue

                    if (
                        self.gradient_asymmetric_dx[py2, px2] == 0.0
                        and self.gradient_asymmetric_dy[py2, px2] == 0.0
                    ):
                        continue


                    position_tolerance_x = (
                        symmetric_position_tolerance / 100.0
                    ) * self.cols

                    position_tolerance_y = (
                        symmetric_position_tolerance / 100.0
                    ) * self.rows

                    if opposite_vector_tolerance is not None:
                        gx1 = self.gradient_asymmetric_dx[py, px]
                        gy1 = self.gradient_asymmetric_dy[py, px]

                        gx2 = self.gradient_asymmetric_dx[py2, px2]
                        gy2 = self.gradient_asymmetric_dy[py2, px2]

                        # IDL criterion: magnitude of the vector sum
                        r = np.sqrt(
                            (gx1 + gx2)**2 +
                            (gy1 + gy2)**2
                        )

                        # Normalize the residual
                        denom = np.sqrt(gx1**2 + gy1**2) + np.sqrt(gx2**2 + gy2**2)

                        if denom > 0:
                            tolerance_error = r / denom
                        else:
                            tolerance_error = 0.0

                        # print(f"Tolerance error: {tolerance_error}")

                        if tolerance_error <= opposite_vector_tolerance:
                            # Check whether the two pixels are opposite
                            # with respect to the analysis center.
                            # (px, py) + (px2, py2) ≈ 2*(cx, cy)

                            position_opposite = (
                                abs((px + px2) - 2 * self.cx) <= position_tolerance_x
                                and
                                abs((py + py2) - 2 * self.cy) <= position_tolerance_y
                            )

                            if position_opposite:

                                self.gradient_asymmetric_dx[py, px] = np.float32(0.0)
                                self.gradient_asymmetric_dy[py, px] = np.float32(0.0)

                                self.gradient_asymmetric_dx[py2, px2] = np.float32(0.0)
                                self.gradient_asymmetric_dy[py2, px2] = np.float32(0.0)

                                break
                    
                    elif abs(self.mods[py, px] - self.mods[py2, px2]) <= magnitude_tolerance * self.maxGrad:
                    
                        # Check whether the gradient vectors have
                        # approximately opposite orientations.
                        angle_opposite = (
                            abs(
                                self._angleDifference(
                                    self.phases[py, px],
                                    self.phases[py2, px2]
                                ) - np.pi
                            ) <= angle_tolerance
                        )

                        # Check whether the two pixels are opposite
                        # with respect to the analysis center.
                        # (px, py) + (px2, py2) ≈ 2*(cx, cy)
                        position_opposite = (
                            abs((px + px2) - 2 * self.cx) <= position_tolerance_x
                            and
                            abs((py + py2) - 2 * self.cy) <= position_tolerance_y
                        )

                        if angle_opposite and position_opposite:

                            self.gradient_asymmetric_dx[py, px] = np.float32(0.0)
                            self.gradient_asymmetric_dy[py, px] = np.float32(0.0)

                            self.gradient_asymmetric_dx[py2, px2] = np.float32(0.0)
                            self.gradient_asymmetric_dy[py2, px2] = np.float32(0.0)

                            break

        # Preserve compatibility with the original Cython implementation.
        if len(removedP) > 0:
            self.removedP = np.array(removedP, dtype=np.int32)

        # Reset the vector counters.
        self.totalVet = 0
        self.totalAssimetric = 0

        nremovedP = []

        # Count the remaining asymmetric vectors.
        for j in range(self.rows):
            for i in range(self.cols):

                # Valid asymmetric gradient vector.
                if (
                    (
                        self.gradient_asymmetric_dy[j, i] != 0.0
                        or self.gradient_asymmetric_dx[j, i] != 0.0
                    )
                    and mask[j, i] != 0.0
                ):
                    nremovedP.append([j, i])
                    self.totalVet += 1

                # Gradient vector removed due to radial symmetry.
                elif mask[j, i] != 0.0:
                    removedP.append([j, i])
                    self.totalVet += 1
                    self.totalAssimetric += 1

        # Store the remaining asymmetric points.
        if len(nremovedP) > 0:
            self.nremovedP = np.array(nremovedP, dtype=np.int32)

    

    def _angleDifference(self, a1, a2):
        diff = abs(a1-a2)
        return min(diff, 2*np.pi-diff)
    

    
    def _G1(self):
        """
        Compute the first-order Gradient Pattern Analysis (G1) index.

        This implementation reproduces in Python the original first-order
        Gradient Pattern Analysis formulation introduced by Rosa, Sharma,
        and Valdivia (1999). The mathematical procedure and normalization
        used in the original implementation are preserved; only the
        programming language has been changed from the original
        implementation to Python.

        The G1 index is calculated from the Delaunay triangulation of the
        non-zero asymmetric gradient vectors. The spatial position of each
        gradient vector is combined with its normalized gradient components
        to define the points used in the triangulation.

        The number of unique Delaunay edges is compared with the total
        number of non-zero asymmetric gradient vectors according to

            G1 = (N_edges - N_asymmetric) / N_asymmetric

        where ``N_edges`` is the number of unique edges in the Delaunay
        triangulation and ``N_asymmetric`` is the number of non-zero
        asymmetric gradient vectors.

        The gradient components are rescaled using the same normalization
        factor as in the original implementation. This scaling affects only
        the spatial representation of the gradient vectors used to construct
        the asymmetric triangulation.

        References
        ----------
        Rosa, R. R., Sharma, A. S., & Valdivia, J. A. (1999).
        "Characterization of Asymmetric Fragmentation Patterns in Spatially
        Extended Systems."
        International Journal of Modern Physics C, 10(1), 147-163.
        https://doi.org/10.1142/S0129183199000103

        Notes
        -----
        This Python implementation is intended to preserve the mathematical
        formulation of the original 1999 method, rather than introduce a new
        definition of the G1 parameter.
        """

        dx = self.gradient_asymmetric_dx
        dy = self.gradient_asymmetric_dy

        # Select non-zero asymmetric gradient vectors
        # Equivalent to:
        #
        # naozero = WHERE((dx NE 0) OR (dy NE 0))

        naozero = np.flatnonzero(
            (dx != 0) | (dy != 0)
        )

        self.totalAssimetric = len(naozero)

        if self.totalAssimetric < 3:
            self.n_edges = 0
            G1 = 0.0
            return G1

        # IDL factor

        factor = 1.0 / (
            2.0 * np.sqrt(
                np.abs(np.max(dx))**2 +
                np.abs(np.max(dy))**2
            )
        )

        # Convert flattened IDL indices to (row, column)

        rows, cols = np.unravel_index(
            naozero,
            dx.shape
        )

        # IDL:
        #
        # vvx = dx(naozero)*factor + naozero MOD my
        # vvy = dy(naozero)*factor + naozero/my

        self.vvx = (
            dx[rows, cols] * factor
            + cols
        )

        self.vvy = (
            dy[rows, cols] * factor
            + rows
        )

        # Delaunay triangulation

        triangulation_points = np.column_stack(
            (self.vvx, self.vvy)
        ).astype(np.float64)

        self.triangles = Delaunay(
            triangulation_points
        )

        # Number of unique Delaunay edges

        indptr, indices = (
            self.triangles.vertex_neighbor_vertices
        )

        self.n_edges = len(indices) / 2.0

        # GPA / Fragmentation

        G1 = (
            self.n_edges - self.totalAssimetric
        ) / self.totalAssimetric

        return G1



    def _G2(self):
        """
        Compute the second Gradient Pattern Analysis (GPA) moment.

        The second GPA moment is defined as

            G2 = (V / VA) · (2 - D)

        where

            V  = number of remaining asymmetric gradient vectors,
            VA = total number of valid gradient vectors, and

            D = |Σvi| / Σ|vi|

        is the vectorial diversity (or alignment) measure. Values of
        ``D`` close to 1 indicate highly aligned gradient vectors,
        whereas smaller values indicate greater directional diversity.

        Returns
        -------
        float
            The second GPA moment (G2).
        """

        # Count the remaining asymmetric gradient vectors.
        if len(self.nremovedP) > 0:
            self.totalAssimetric = len(self.nremovedP[:, 0])
        else:
            self.totalAssimetric = 0

        # Compute the vectorial diversity:
        #
        # D = |Σvi| / Σ|vi|
        self.phaseDiversity = self._vectorialVariety()

        # Compute the second GPA moment:
        #
        # G2 = (V / VA) · (2 - D)
        G2 = (
            float(self.totalAssimetric) / float(self.totalVet)
        ) * (2.0 - self.phaseDiversity)

        return G2


    
    def _G3(self):
            
        return
    def _G4(self):
            
        return



    def _vectorialVariety(self):
        """
        Compute the vectorial diversity (alignment) measure.

        The quantity is defined as

            D = |Σvi| / Σ|vi|

        where

            vi = (Gx, Gy)

        is an asymmetric gradient vector.

        This measure quantifies the overall alignment of the asymmetric
        gradient vectors. Values close to 1 indicate that the vectors are
        predominantly aligned in the same direction, whereas values close
        to 0 indicate a more diverse or isotropic distribution of
        orientations.

        Returns
        -------
        float
            The vectorial diversity (alignment) measure.
        """

        sum_x = 0.0
        sum_y = 0.0
        sum_magnitude = 0.0

        # No asymmetric vectors are available.
        if self.totalAssimetric < 1:
            return 0.0

        # Sum the asymmetric gradient vectors.
        for i in range(self.totalAssimetric):

            row = self.nremovedP[i, 0]
            col = self.nremovedP[i, 1]

            # Gradient magnitude:
            #
            # |vi| = sqrt(Gx² + Gy²)
            magnitude = self.mods[row, col]

            sum_x += self.gradient_dx[row, col]
            sum_y += self.gradient_dy[row, col]

            # Sum of the vector magnitudes.
            sum_magnitude += magnitude

        if sum_magnitude <= 0.0:
            return 0.0

        # Compute the vectorial diversity:
        #
        # D = sqrt((ΣGx)² + (ΣGy)²) / Σ|vi|
        vectorial_diversity = (
            np.sqrt(sum_x**2 + sum_y**2) / sum_magnitude
        )

        return vectorial_diversity



    def plot_delaunay_triangulation(
        self,
        show_image=True,
        show_scale=True,
        show_center=True,
        line_color="red"
    ):
        """
        Plot the asymmetric gradient field and its Delaunay triangulation.

        The triangulation is constructed from the positions of the
        remaining asymmetric gradient vectors.

        Parameters
        ----------
        show_image : bool, default=True
            If True, display the original matrix as the background.

        show_scale : bool, default=True
            If True, display the x and y axes, including tick marks and
            labels. If False, hide the ticks and labels while preserving
            the plot area.

        show_center : bool, default=True
            If True, display the analysis center.

        line_color : str, default="red"
            Color of the Delaunay triangulation lines.
        """

        gx = self.gradient_asymmetric_dx
        gy = self.gradient_asymmetric_dy

        valid = (
            self.mask.astype(bool)
            & ((gx != 0) | (gy != 0))
        )

        has_vectors = np.any(valid)

        # Pixel-centered coordinates
        y, x = np.mgrid[
            0:self.rows,
            0:self.cols
        ]

        x = x + 0.5
        y = y + 0.5

        # Determine plotting region

        if max(self.rows, self.cols) > 50 and has_vectors:

            rows, cols = np.where(valid)

            margin = 2

            ymin = max(rows.min() - margin, 0)
            ymax = min(rows.max() + margin + 1, self.rows)

            xmin = max(cols.min() - margin, 0)
            xmax = min(cols.max() + margin + 1, self.cols)

        else:

            ymin = 0
            ymax = self.rows

            xmin = 0
            xmax = self.cols

        
        # No asymmetric vectors

        if not has_vectors:

            print("No asymmetric gradient vectors remaining.")
            return

        # Figure

        fig, ax = plt.subplots(figsize=(5, 5))

        # Background image

        if show_image:

            ax.imshow(
                self.matrix[ymin:ymax, xmin:xmax],
                cmap="gray",
                origin="lower",
                extent=[
                    xmin,
                    xmax,
                    ymin,
                    ymax
                ]
            )

        # Normalize vectors for visualization

        max_gx = np.max(np.abs(gx))
        max_gy = np.max(np.abs(gy))

        factor = 1.0 / (
            2.0 * np.sqrt(
                max_gx**2 +
                max_gy**2
            )
        )

        u = gx * factor
        v = gy * factor

        # Asymmetric gradient vectors

        ax.quiver(
            x[valid],
            y[valid],
            u[valid],
            v[valid],
            color=line_color,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003
        )

        # Delaunay triangulation

        ax.triplot(
            self.vvx + 0.5,
            self.vvy + 0.5,
            self.triangles.simplices,
            color=line_color,
            linewidth=0.8
        )

        # Triangulation points

        ax.scatter(
            self.vvx + 0.5,
            self.vvy + 0.5,
            s=8,
            color=line_color,
            zorder=3
        )

        # Analysis center

        if show_center:

            ax.scatter(
                self.cx + 0.5,
                self.cy + 0.5,
                marker="x",
                color="blue",
                s=100,
                linewidths=2,
                zorder=4
            )

        # Plot limits

        margin = 0.5

        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)

        # Minimal style

        if show_scale:

            ax.set_xlabel("x")
            ax.set_ylabel("y")

        else:

            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_xlabel("")
            ax.set_ylabel("")

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)

        ax.set_aspect("equal")

        plt.tight_layout(pad=0)

        plt.show()

    def plot_gradient_field(
        self,
        fixed_length=True,
        show_image=True,
        show_scale=True,
        show_center=True,
        color="red"
    ):
        """
        Plot the image gradient field as a vector field.

        Each arrow represents a gradient vector:

            ∇I = (Gx, Gy)

        Parameters
        ----------
        fixed_length : bool, default=True
            If True, normalize all vectors to the same length so that
            only their orientations are represented. If False, preserve
            the relative gradient magnitudes.

        show_image : bool, default=True
            If True, display the original matrix as the background of
            the gradient field.

        show_scale : bool, default=True
            If True, display the x and y axes, including tick marks and
            labels. If False, hide the ticks and labels while preserving
            the plot area.

        show_center : bool, default=True
            If True, display the analysis center.

        color : str, default="red"
            Color of the gradient vectors.
        """

        # Pixels where vectors will be displayed

        valid = self.mask.astype(bool)

        gx = self.gradient_dx
        gy = self.gradient_dy

        # Coordinate grid

        y, x = np.mgrid[
            0:self.rows,
            0:self.cols
        ]

        x = x + 0.5
        y = y + 0.5

        # Normalize vectors or preserve magnitudes

        if fixed_length:

            magnitude = np.sqrt(gx**2 + gy**2)

            # Avoid division by zero
            magnitude[magnitude == 0] = 1

            scale = 0.5

            u = (gx / magnitude) * scale
            v = (gy / magnitude) * scale

        else:

            max_gx = np.max(np.abs(gx))
            max_gy = np.max(np.abs(gy))

            factor = 1.0 / (
                2.0 * np.sqrt(
                    max_gx**2 +
                    max_gy**2
                )
            )

            u = gx * factor
            v = gy * factor

        # Bounding box of the mask

        rows, cols = np.where(valid)

        margin = 2

        ymin = max(rows.min() - margin, 0)
        ymax = min(rows.max() + margin + 1, self.rows)

        xmin = max(cols.min() - margin, 0)
        xmax = min(cols.max() + margin + 1, self.cols)

        # Figure

        fig, ax = plt.subplots(figsize=(5, 5))

        # Background image

        if show_image:

            ax.imshow(
                self.matrix[ymin:ymax, xmin:xmax],
                cmap="gray",
                origin="lower",
                extent=[
                    xmin,
                    xmax,
                    ymin,
                    ymax
                ]
            )

        # Gradient vectors

        ax.quiver(
            x[valid],
            y[valid],
            u[valid],
            v[valid],
            color=color,
            angles="xy",
            scale_units="xy",
            scale=1
        )

        # Analysis center

        if show_center:

            ax.scatter(
                self.cx + 0.5,
                self.cy + 0.5,
                marker="x",
                color="blue",
                s=150,
                linewidths=2
            )

        # Plot limits

        margin = 0.5

        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)

        # Minimal style

        if show_scale:

            ax.set_xlabel("x")
            ax.set_ylabel("y")

        else:

            # Remove ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_xlabel("")
            ax.set_ylabel("")

            # Keep the boundary of the image/plot
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)

        ax.set_aspect("equal")

        plt.tight_layout(pad=0)

        plt.show()

    def plot_asymmetric_gradient_field(
        self,
        fixed_length=True,
        show_image=True,
        show_scale=True,
        show_center=True,
        color="red"
    ):
        """
        Plot the asymmetric gradient field obtained after removing
        symmetric gradient contributions.

        Parameters
        ----------
        fixed_length : bool, default=True
            If True, normalize all vectors to the same length so that
            only their orientations are represented. If False, preserve
            the relative gradient magnitudes.

        show_image : bool, default=True
            If True, display the original matrix as the background of
            the gradient field.

        show_scale : bool, default=True
            If True, display the x and y axes, including tick marks and
            labels. If False, hide the ticks and labels while preserving
            the plot area.

        show_center : bool, default=True
            If True, display the analysis center used to identify
            symmetric gradient contributions.

        color : str, default="red"
            Color of the asymmetric gradient vectors.
        """

        gx = self.gradient_asymmetric_dx
        gy = self.gradient_asymmetric_dy

        valid = (
            self.mask.astype(bool)
            & ((gx != 0) | (gy != 0))
        )

        has_vectors = np.any(valid)

        # Coordinate grid

        y, x = np.mgrid[
            0:self.rows,
            0:self.cols
        ]

        x = x + 0.5
        y = y + 0.5

        # Determine plotting region

        if max(self.rows, self.cols) > 50 and has_vectors:

            rows, cols = np.where(valid)

            margin = 2

            ymin = max(rows.min() - margin, 0)
            ymax = min(rows.max() + margin + 1, self.rows)

            xmin = max(cols.min() - margin, 0)
            xmax = min(cols.max() + margin + 1, self.cols)

        else:

            ymin = 0
            ymax = self.rows

            xmin = 0
            xmax = self.cols

        # Normalize vectors if requested

        if has_vectors:
            
            if fixed_length:

                magnitude = np.sqrt(gx**2 + gy**2)

                # Avoid division by zero
                magnitude[magnitude == 0] = 1

                scale = 0.5

                u = (gx / magnitude) * scale
                v = (gy / magnitude) * scale

            else:

                max_gx = np.max(np.abs(gx))
                max_gy = np.max(np.abs(gy))

                factor = 1.0 / (
                    2.0 * np.sqrt(
                        max_gx**2 +
                        max_gy**2
                    )
                )

                u = gx * factor
                v = gy * factor

        else:

            print("No asymmetric gradient vectors remaining.")
            return

        # Figure

        fig, ax = plt.subplots(figsize=(5, 5))

        # Background image

        if show_image:

            ax.imshow(
                self.matrix[ymin:ymax, xmin:xmax],
                cmap="gray",
                origin="lower",
                extent=[
                    xmin,
                    xmax,
                    ymin,
                    ymax
                ]
            )

        # Asymmetric gradient vectors

        if has_vectors:

            ax.quiver(
                x[valid],
                y[valid],
                u[valid],
                v[valid],
                color=color,
                angles="xy",
                scale_units="xy",
                scale=1
            )

        # Analysis center

        if show_center:

            ax.scatter(
                self.cx + 0.5,
                self.cy + 0.5,
                marker="x",
                color="blue",
                s=150,
                linewidths=2,
                zorder=4
            )

        # Plot limits

        margin = 0.5

        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)

        # Minimal style

        if show_scale:

            ax.set_xlabel("x")
            ax.set_ylabel("y")

        else:

            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_xlabel("")
            ax.set_ylabel("")

            # Keep the plot boundary
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)

        ax.set_aspect("equal")

        plt.tight_layout(pad=0)

        plt.show()