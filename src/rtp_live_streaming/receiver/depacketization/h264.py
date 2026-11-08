"""H.264 RTP depacketization according to RFC 6184.

Implements packetization mode 1:
    - Single NAL unit packets (types 1-23).
    - Fragmentation Units (FU-A, type 28).

The depacketizer reconstructs complete NAL units and groups them
into access units using the RTP marker bit.

Input:
    RTP payload bytes and the RTP marker bit.

Output:
    A tuple of complete NAL units when an access unit is complete.
    None when more RTP packets are needed.

NAL units are returned without Annex-B start codes.
"""

from dataclasses import dataclass, field

NAL_HEADER_SIZE = 1
FU_A_HEADER_SIZE = 2

NAL_TYPE_MIN = 1
NAL_TYPE_MAX = 23
NAL_TYPE_FU_A = 28

FORBIDDEN_ZERO_BIT = 0x80
NAL_REF_IDC_MASK = 0x60
NAL_TYPE_MASK = 0x1F

FU_START_BIT = 0x80
FU_END_BIT = 0x40
FU_RESERVED_BIT = 0x20


class H264DepacketizationError(ValueError):
    """Raised when an H.264 RTP payload is invalid or incomplete."""


@dataclass(slots=True)
class H264Depacketizer:
    """Reassemble H.264 NAL units from RTP payloads.

    Call push() for each RTP packet in sequence-number order.

    An access unit is returned when a packet with marker=True arrives.
    Each returned item is one complete NAL unit.
    """

    _current_nal: bytearray | None = field(default=None, init=False)
    _access_unit: list[bytes] = field(default_factory=list, init=False)

    def push(
        self,
        payload: bytes,
        marker: bool = False,
    ) -> tuple[bytes, ...] | None:
        """Process one H.264 RTP payload.

        Args:
            payload: RTP payload bytes.
            marker: RTP marker bit.

        Returns:
            A tuple of reconstructed NAL units when the access unit
            is complete; otherwise None.

        Raises:
            H264DepacketizationError: If the payload is malformed or
                fragments arrive in an invalid order.
        """
        if not isinstance(payload, bytes):
            raise H264DepacketizationError("Payload must be bytes")

        if not payload:
            raise H264DepacketizationError("Payload cannot be empty")

        if not isinstance(marker, bool):
            raise H264DepacketizationError("marker must be a boolean")

        nal_type = payload[0] & NAL_TYPE_MASK

        try:
            if NAL_TYPE_MIN <= nal_type <= NAL_TYPE_MAX:
                self._process_single_nal(payload)

            elif nal_type == NAL_TYPE_FU_A:
                self._process_fu_a(payload)

            else:
                raise H264DepacketizationError(
                    f"Unsupported H.264 RTP payload type: {nal_type}"
                )

            if marker:
                return self._finish_access_unit()

            return None

        except H264DepacketizationError:
            self.reset()
            raise

    def _process_single_nal(self, payload: bytes) -> None:
        """Process an RTP payload containing one complete NAL unit."""
        if self._current_nal is not None:
            raise H264DepacketizationError(
                "Received a single NAL unit while FU-A reassembly was still in progress"
            )

        header = payload[0]

        if header & FORBIDDEN_ZERO_BIT:
            raise H264DepacketizationError("NAL unit forbidden_zero_bit must be zero")

        self._access_unit.append(payload)

    def _process_fu_a(self, payload: bytes) -> None:
        """Process and reassemble one FU-A fragment."""
        if len(payload) <= FU_A_HEADER_SIZE:
            raise H264DepacketizationError("FU-A payload must contain fragment data")

        fu_indicator = payload[0]
        fu_header = payload[1]

        if fu_indicator & FORBIDDEN_ZERO_BIT:
            raise H264DepacketizationError("FU-A forbidden_zero_bit must be zero")

        if fu_header & FU_RESERVED_BIT:
            raise H264DepacketizationError("FU-A reserved bit must be zero")

        is_start = bool(fu_header & FU_START_BIT)
        is_end = bool(fu_header & FU_END_BIT)

        if is_start and is_end:
            raise H264DepacketizationError("FU-A start and end bits cannot both be set")

        nal_type = fu_header & NAL_TYPE_MASK

        if not NAL_TYPE_MIN <= nal_type <= NAL_TYPE_MAX:
            raise H264DepacketizationError(
                f"Invalid FU-A original NAL type: {nal_type}"
            )

        fragment_data = payload[FU_A_HEADER_SIZE:]

        if is_start:
            if self._current_nal is not None:
                raise H264DepacketizationError(
                    "Received a new FU-A start fragment before "
                    "the previous NAL unit was completed"
                )

            # Reconstruct the original NAL header from the FU indicator
            # and the original NAL type stored in the FU header.
            nal_header = (
                fu_indicator & (FORBIDDEN_ZERO_BIT | NAL_REF_IDC_MASK)
            ) | nal_type

            self._current_nal = bytearray((nal_header,))
            self._current_nal.extend(fragment_data)
            return

        if self._current_nal is None:
            raise H264DepacketizationError(
                "Received a FU-A continuation/end fragment without a start fragment"
            )

        # FU indicator F and NRI fields must remain consistent
        # throughout the fragmented NAL unit.
        expected_indicator = (
            self._current_nal[0] & (FORBIDDEN_ZERO_BIT | NAL_REF_IDC_MASK)
        ) | NAL_TYPE_FU_A

        if fu_indicator != expected_indicator:
            raise H264DepacketizationError(
                "FU-A indicator changed during NAL reassembly"
            )

        if (self._current_nal[0] & NAL_TYPE_MASK) != nal_type:
            raise H264DepacketizationError(
                "FU-A original NAL type changed during reassembly"
            )

        self._current_nal.extend(fragment_data)

        if is_end:
            self._access_unit.append(bytes(self._current_nal))
            self._current_nal = None

    def _finish_access_unit(self) -> tuple[bytes, ...]:
        """Return all NAL units belonging to the completed access unit."""
        if self._current_nal is not None:
            raise H264DepacketizationError(
                "Access unit ended before FU-A reassembly completed"
            )

        if not self._access_unit:
            raise H264DepacketizationError("Access unit contains no NAL units")

        completed = tuple(self._access_unit)
        self._access_unit.clear()

        return completed

    def reset(self) -> None:
        """Discard any incomplete NAL unit and access unit."""
        self._current_nal = None
        self._access_unit.clear()
