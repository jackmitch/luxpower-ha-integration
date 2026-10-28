"""The Inverter Time sensor, built from hold registers 12-14.

An inverter's clock is plain wall-clock time with no timezone, and its charge and
discharge windows follow it. After the UK clocks went back an inverter that had been
left on BST ran an hour fast, and Predbat started charging 30 minutes before the cheap
window without anyone being told. Predbat's inverter_time setting needs a sensor to read, and
this is it.
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity_component import EntityComponent
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.lxp_modbus.const import DOMAIN
from custom_components.lxp_modbus.constants.hold_registers import (
    H_TIME_YEAR_MONTH, H_TIME_DATE_HOUR, H_TIME_MINUTE_SECOND,
)
from custom_components.lxp_modbus.entity_descriptions.sensor_types import SENSOR_TYPES
from custom_components.lxp_modbus.sensor import (
    ModbusBridgeClockSensor, ModbusBridgeSensor, _sensor_class,
)
from custom_components.lxp_modbus.utils import decode_inverter_clock

ENTRY_ID = "test_entry_id"
LONDON = ZoneInfo("Europe/London")

# A real reading from a LuxPower inverter: 2026-10-27 09:45:59
REAL_READING = {H_TIME_YEAR_MONTH: 2586, H_TIME_DATE_HOUR: 2331, H_TIME_MINUTE_SECOND: 15149}


def pack(moment: datetime) -> dict:
    """Pack a datetime into the three clock registers (two fields each, low byte first)."""
    return {
        H_TIME_YEAR_MONTH: (moment.year - 2000) | (moment.month << 8),
        H_TIME_DATE_HOUR: moment.day | (moment.hour << 8),
        H_TIME_MINUTE_SECOND: moment.minute | (moment.second << 8),
    }


class TestDecodeInverterClock:
    def test_decodes_a_real_reading(self):
        assert decode_inverter_clock(REAL_READING) == datetime(2026, 10, 27, 9, 45, 59)

    def test_round_trips_every_field(self):
        moment = datetime(2031, 12, 31, 23, 59, 58)

        assert decode_inverter_clock(pack(moment)) == moment

    def test_result_is_naive(self):
        """The inverter reports wall-clock time with no zone of its own."""
        assert decode_inverter_clock(REAL_READING).tzinfo is None

    @pytest.mark.parametrize("missing", [H_TIME_YEAR_MONTH, H_TIME_DATE_HOUR, H_TIME_MINUTE_SECOND])
    def test_a_missing_register_gives_none(self, missing):
        """A poll that returned some register blocks and not others must not make a time."""
        registers = {k: v for k, v in REAL_READING.items() if k != missing}

        assert decode_inverter_clock(registers) is None

    def test_no_hold_data_gives_none(self):
        assert decode_inverter_clock({}) is None

    def test_all_zero_registers_give_none(self):
        """Month 0 and day 0 are not a date, which is what an unset inverter reports."""
        assert decode_inverter_clock(
            {H_TIME_YEAR_MONTH: 0, H_TIME_DATE_HOUR: 0, H_TIME_MINUTE_SECOND: 0}
        ) is None

    @pytest.mark.parametrize("year_month,date_hour,minute_second", [
        (26 | (13 << 8), 1 | (0 << 8), 0),      # month 13
        (26 | (2 << 8), 30 | (0 << 8), 0),       # 30 February
        (26 | (10 << 8), 27 | (24 << 8), 0),     # hour 24
        (26 | (10 << 8), 27 | (9 << 8), 60),     # minute 60
        (26 | (10 << 8), 27 | (9 << 8), 60 << 8),  # second 60
    ])
    def test_impossible_dates_give_none(self, year_month, date_hour, minute_second):
        registers = {
            H_TIME_YEAR_MONTH: year_month,
            H_TIME_DATE_HOUR: date_hour,
            H_TIME_MINUTE_SECOND: minute_second,
        }

        assert decode_inverter_clock(registers) is None

    def test_leap_day(self):
        assert decode_inverter_clock(pack(datetime(2028, 2, 29, 12, 0, 0))) == datetime(2028, 2, 29, 12, 0, 0)
        assert decode_inverter_clock(
            {H_TIME_YEAR_MONTH: 27 | (2 << 8), H_TIME_DATE_HOUR: 29, H_TIME_MINUTE_SECOND: 0}
        ) is None   # 2027 is not a leap year


class TestDescription:
    def _description(self):
        matches = [d for d in SENSOR_TYPES if d["name"] == "Inverter Time"]
        assert len(matches) == 1, f"expected one 'Inverter Time', found {len(matches)}"
        return matches[0]

    def test_depends_on_the_three_clock_registers(self):
        assert self._description()["depends_on"] == [12, 13, 14]

    def test_is_a_timestamp_sensor(self):
        desc = self._description()

        assert desc["device_class"] == "timestamp"
        assert desc["extract_clock"] is decode_inverter_clock

    def test_gets_the_clock_sensor_class(self):
        """Every other sensor keeps the base class, so nothing else changes."""
        assert _sensor_class(self._description()) is ModbusBridgeClockSensor
        others = [d for d in SENSOR_TYPES if d["name"] != "Inverter Time"]
        assert all(_sensor_class(d) is ModbusBridgeSensor for d in others)

    def test_enabled_by_default(self):
        """A skewed clock is what you want to be told about, and a disabled entity is hidden
        from the entity list by default, so few would ever find it. It changes at most once
        a minute, which costs less than the busiest existing sensors."""
        assert self._description()["enabled"] is True


class TestSensorValue:
    @pytest.fixture(autouse=True)
    def london_zone(self):
        previous = dt_util.get_default_time_zone()
        dt_util.set_default_time_zone(LONDON)
        yield
        dt_util.set_default_time_zone(previous)

    @pytest.fixture
    def coordinator(self):
        coord = MagicMock()
        coord.async_request_refresh = AsyncMock()
        coord.hass = MagicMock()
        coord.hass.data = {DOMAIN: {ENTRY_ID: {"write_lock": MagicMock()}}}
        coord.data = {"input": {}, "hold": dict(REAL_READING)}
        return coord

    @pytest.fixture
    def entry(self):
        cfg = MagicMock()
        cfg.entry_id = ENTRY_ID
        cfg.title = "Test Inverter"
        cfg.data = {}
        return cfg

    def _sensor(self, coordinator, entry):
        desc = next(d for d in SENSOR_TYPES if d["name"] == "Inverter Time")
        return ModbusBridgeClockSensor(coordinator, entry, dict(desc), "lxp", None)

    def test_value_is_aware_in_the_home_assistant_zone(self, coordinator, entry):
        value = self._sensor(coordinator, entry).native_value

        # 09:45:59 to the minute
        assert value == datetime(2026, 10, 27, 9, 46, 0, tzinfo=LONDON)
        assert value.utcoffset() == timedelta(0)         # GMT after the clocks went back

    def test_summer_time_offset_is_applied(self, coordinator, entry):
        coordinator.data["hold"] = pack(datetime(2026, 6, 15, 12, 0, 0))

        value = self._sensor(coordinator, entry).native_value

        assert value.utcoffset() == timedelta(hours=1)
        assert value.isoformat() == "2026-06-15T12:00:00+01:00"

    def test_an_inverter_still_on_bst_reads_an_hour_ahead(self, coordinator, entry):
        """The failure this sensor exists to show: the real time is 08:45 GMT."""
        real_now = datetime(2026, 10, 27, 8, 45, 59, tzinfo=LONDON)

        value = self._sensor(coordinator, entry).native_value

        skew = value - real_now
        assert abs(skew - timedelta(hours=1)) <= timedelta(seconds=30)    # to the minute

    @pytest.mark.parametrize("seconds,expected_minute", [
        (0, 45), (29, 45), (30, 46), (59, 46),      # to the nearest minute
    ])
    def test_value_is_to_the_minute(self, coordinator, entry, seconds, expected_minute):
        """Predbat's thresholds are minutes; a state that moves every second is just recorder load."""
        coordinator.data["hold"] = pack(datetime(2026, 10, 27, 9, 45, seconds))

        value = self._sensor(coordinator, entry).native_value

        assert (value.minute, value.second, value.microsecond) == (expected_minute, 0, 0)

    def test_clock_state_is_kept_per_sensor(self, coordinator, entry, freezer):
        """The advance between settings reads is kept on each sensor, not shared.

        Two sensors with their own coordinators must neither move nor stall each other.
        """
        freezer.move_to("2026-10-27T09:45:59+00:00")
        other_coordinator = MagicMock()
        other_coordinator.hass = coordinator.hass
        other_coordinator.data = {"input": {}, "hold": pack(datetime(2026, 10, 27, 11, 0, 0))}   # 2 h ahead
        first = self._sensor(coordinator, entry)
        second = self._sensor(other_coordinator, entry)
        assert first.native_value == datetime(2026, 10, 27, 9, 46, 0, tzinfo=LONDON)
        assert second.native_value == datetime(2026, 10, 27, 11, 0, 0, tzinfo=LONDON)

        freezer.tick(timedelta(minutes=4))          # four minutes pass with no settings read

        assert first.native_value == datetime(2026, 10, 27, 9, 50, 0, tzinfo=LONDON)
        assert second.native_value == datetime(2026, 10, 27, 11, 4, 0, tzinfo=LONDON)

    def test_every_poll_is_written(self, coordinator, entry):
        """No update-filter fingerprint, or the clock freezes.

        depends_on names hold registers 12-14, but the filter looks a calculated
        sensor's dependencies up in the input registers, where 12-14 are the grid
        voltages. With a fingerprint the clock only updated when the grid voltage did.
        """
        assert self._sensor(coordinator, entry)._source_fingerprint() is None

    @pytest.mark.parametrize("data", [None, {}, {"hold": {}}, {"input": {}}])
    def test_unknown_until_the_clock_has_been_read(self, coordinator, entry, data):
        coordinator.data = data

        assert self._sensor(coordinator, entry).native_value is None

    def test_unique_id_is_stable(self, coordinator, entry):
        sensor = self._sensor(coordinator, entry)

        assert sensor._attr_unique_id == "lxp_12_13_14_inverter_time"
        assert sensor._attr_entity_registry_enabled_default is True


class TestInHomeAssistant:
    """The sensor added to a real Home Assistant, driven by coordinator updates.

    Calling native_value directly cannot see how often the state is written, or what
    state string Home Assistant publishes, and the first version of this sensor had
    bugs in both.
    """

    GRID_VOLTAGES = {12: 2400, 13: 0, 14: 0}    # the same numbers, as input registers

    async def _add_sensor(self, hass):
        await hass.config.async_set_time_zone("Europe/London")
        entry = MockConfigEntry(domain=DOMAIN, data={}, title="Inverter")
        entry.add_to_hass(hass)
        hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {"write_lock": None}

        coordinator = DataUpdateCoordinator(
            hass, logging.getLogger(__name__), config_entry=entry, name="test")
        coordinator.data = {"input": dict(self.GRID_VOLTAGES), "hold": dict(REAL_READING)}

        desc = dict(next(d for d in SENSOR_TYPES if d["name"] == "Inverter Time"))
        sensor = ModbusBridgeClockSensor(coordinator, entry, desc, "lxp", None)
        await EntityComponent(logging.getLogger(__name__), "sensor", hass).async_add_entities([sensor])
        await hass.async_block_till_done()
        return coordinator, sensor

    async def _poll(self, hass, coordinator, **changes):
        coordinator.async_set_updated_data({**coordinator.data, **changes})
        await hass.async_block_till_done()

    async def test_published_state_parses_with_predbats_time_format(self, hass):
        """Predbat parses '%Y-%m-%dT%H:%M:%S%z', and there is no unit to trip a non-numeric class."""
        _, sensor = await self._add_sensor(hass)

        state = hass.states.get(sensor.entity_id)

        assert state.state == "2026-10-27T09:46:00+00:00"
        assert "unit_of_measurement" not in state.attributes
        assert datetime.strptime(state.state, "%Y-%m-%dT%H:%M:%S%z") == datetime(
            2026, 10, 27, 9, 46, 0, tzinfo=ZoneInfo("UTC"))

    async def test_new_clock_reading_is_published_when_grid_voltage_is_unchanged(self, hass):
        """The grid voltage is what registers 12-14 mean as input registers.

        It is constant when the grid is down, and the clock must keep moving then.
        """
        coordinator, sensor = await self._add_sensor(hass)

        for minute in (50, 55):
            await self._poll(hass, coordinator, hold=pack(datetime(2026, 10, 27, 9, minute, 59)))

        assert hass.states.get(sensor.entity_id).state == "2026-10-27T09:56:00+00:00"

    async def test_clock_advances_between_settings_reads(self, hass, freezer):
        """Settings registers are re-read every 5th poll; the clock must not look 4 minutes slow."""
        freezer.move_to("2026-10-27T09:45:59+00:00")
        coordinator, sensor = await self._add_sensor(hass)

        freezer.tick(timedelta(seconds=240))
        await self._poll(hass, coordinator)          # an input-only poll: hold data unchanged

        assert hass.states.get(sensor.entity_id).state == "2026-10-27T09:50:00+00:00"

    async def test_a_fresh_settings_read_replaces_the_advanced_clock(self, hass, freezer):
        freezer.move_to("2026-10-27T09:45:59+00:00")
        coordinator, sensor = await self._add_sensor(hass)
        freezer.tick(timedelta(seconds=240))
        await self._poll(hass, coordinator)

        # The settings read shows the inverter an hour ahead of the real 09:49:59. The new
        # reading replaces the advanced one, so the skew is visible and is not carried over.
        await self._poll(hass, coordinator, hold=pack(datetime(2026, 10, 27, 10, 49, 59)))

        assert hass.states.get(sensor.entity_id).state == "2026-10-27T10:50:00+00:00"

    async def test_state_changes_once_a_minute_however_often_it_polls(self, hass, freezer):
        """The point of minute resolution: 15 second polls must not mean a row every 15 seconds."""
        freezer.move_to("2026-10-27T09:45:59+00:00")
        coordinator, sensor = await self._add_sensor(hass)
        changes = []
        hass.bus.async_listen(
            "state_changed",
            lambda event: changes.append(event.data["new_state"].state)
            if event.data["entity_id"] == sensor.entity_id else None,
        )

        polls = 8                                      # two minutes of 15 second polls
        for _ in range(polls):
            freezer.tick(timedelta(seconds=15))
            await self._poll(hass, coordinator)

        assert changes == ["2026-10-27T09:47:00+00:00", "2026-10-27T09:48:00+00:00"]
        assert len(changes) < polls

    async def test_repeated_hour_when_the_clocks_go_back(self, hass, freezer):
        """01:30 happens twice on 25 Oct 2026. A correct clock at the second one is 01:30 GMT.

        Reading it as the first (BST) would make it an hour slow, past Predbat's
        30 minute error threshold, for the whole hour.
        """
        freezer.move_to("2026-10-25T01:30:00+00:00")
        coordinator, sensor = await self._add_sensor(hass)

        await self._poll(hass, coordinator, hold=pack(datetime(2026, 10, 25, 1, 30, 0)))

        assert hass.states.get(sensor.entity_id).state == "2026-10-25T01:30:00+00:00"

    async def test_enabled_by_default_in_the_entity_registry(self, hass):
        _, sensor = await self._add_sensor(hass)

        entry = er.async_get(hass).async_get(sensor.entity_id)

        assert entry.disabled_by is None
        assert hass.states.get(sensor.entity_id) is not None
