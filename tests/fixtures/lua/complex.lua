-- Complex Lua Module Example
--
-- Module: Calc Sample
-- Sample module demonstrating complex Lua patterns typical of game modding.
--
local calcs = ...

-- Local aliases for performance
local pairs = pairs
local ipairs = ipairs
local unpack = unpack
local t_insert = table.insert
local t_remove = table.remove
local m_abs = math.abs
local m_floor = math.floor
local m_ceil = math.ceil
local m_min = math.min
local m_max = math.max
local m_sqrt = math.sqrt
local m_pow = math.pow
local m_huge = math.huge
local bor = bit.bor
local band = bit.band
local bnot = bit.bnot
local s_format = string.format

-- Temporary tables for calculations
local tempTable1 = { }
local tempTable2 = { }

-- Element type flags
local isElemental = { Fire = true, Cold = true, Lightning = true }

-- List of all damage types, ordered according to the conversion sequence
local dmgTypeList = {"Physical", "Lightning", "Cold", "Fire", "Chaos"}
local dmgTypeFlags = {
	order = { "Physical", "Lightning", "Cold", "Fire", "Elemental", "Chaos" },
	flags = {
		Physical	= 0x01,
		Lightning	= 0x02,
		Cold		= 0x04,
		Fire		= 0x08,
		Elemental	= 0x0E,
		Chaos		= 0x10,
	}
}

-- Ailment configuration table
local ailmentData = {
	Ignite = {
		damageType = "Fire",
		durationBase = 4,
		stackable = false,
	},
	Bleed = {
		damageType = "Physical",
		durationBase = 5,
		stackable = false,
	},
	Poison = {
		damageType = "Chaos",
		durationBase = 2,
		stackable = true,
	},
}

-- Magic table with __index metamethod for caching
local damageStatsForTypes = setmetatable({ }, { __index = function(t, k)
	local modNames = { "Damage" }
	for _, type in ipairs(dmgTypeFlags.order) do
		local flag = dmgTypeFlags.flags[type]
		if band(k, flag) ~= 0 then
			t_insert(modNames, type.."Damage")
		end
	end
	t[k] = modNames
	return modNames
end })

-- Global output references (common game modding pattern)
local globalOutput = nil
local globalBreakdown = nil

-- Calculate min/max damage for the given damage type
local function calcDamage(activeSkill, output, cfg, breakdown, damageType, typeFlags, convDst)
	local skillModList = activeSkill.skillModList

	typeFlags = bor(typeFlags, dmgTypeFlags.flags[damageType])

	-- Calculate conversions
	local addMin, addMax = 0, 0
	local conversionTable = (cfg and cfg.conversionTable) or activeSkill.conversionTable

	for _, otherType in ipairs(dmgTypeList) do
		if otherType == damageType then
			break
		end
		local convMult = conversionTable[otherType][damageType]
		if convMult > 0 then
			-- Damage is being converted from the other damage type
			local min, max = calcDamage(activeSkill, output, cfg, breakdown, otherType, typeFlags, damageType)
			addMin = addMin + min * convMult
			addMax = addMax + max * convMult
		end
	end

	local baseMin = output[damageType.."MinBase"] or 0
	local baseMax = output[damageType.."MaxBase"] or 0

	-- Apply modifiers
	local inc = 1 + skillModList:Sum("INC", cfg, "Damage", damageType.."Damage") / 100
	local more = skillModList:More(cfg, "Damage", damageType.."Damage")

	local min = (baseMin + addMin) * inc * more
	local max = (baseMax + addMax) * inc * more

	if breakdown then
		breakdown[damageType] = {
			baseMin = baseMin,
			baseMax = baseMax,
			inc = inc,
			more = more,
		}
	end

	return m_floor(min), m_floor(max)
end

-- Calculate ailment damage
local function calcAilmentDamage(activeSkill, output, cfg, ailmentType)
	local ailmentInfo = ailmentData[ailmentType]
	if not ailmentInfo then
		return 0
	end

	local skillModList = activeSkill.skillModList
	local dotMulti = skillModList:Sum("BASE", cfg, ailmentType.."DotMultiplier")
	local damageType = ailmentInfo.damageType

	-- Get base damage
	local baseDamage = output[damageType.."Average"] or 0

	-- Calculate duration
	local duration = ailmentInfo.durationBase
	duration = duration * (1 + skillModList:Sum("INC", cfg, "Duration", ailmentType.."Duration") / 100)
	duration = duration * skillModList:More(cfg, "Duration", ailmentType.."Duration")

	-- Calculate DPS
	local dps = baseDamage * (1 + dotMulti / 100)
	local totalDamage = dps * duration

	return dps, duration, totalDamage
end

-- Merge two modDB entries
local function mergeModDB(modDB, newMods)
	for _, mod in ipairs(newMods) do
		modDB:AddMod(mod)
	end
end

-- Check if a skill has a specific flag
local function skillHasFlag(activeSkill, flag)
	return activeSkill.skillFlags[flag] == true
end

-- Build output table for display
local function buildOutput(actor, mode, env)
	local output = { }
	local breakdown = { }

	output.Mode = mode
	output.Actor = actor.name or "Unknown"

	-- Calculate main hand damage
	if actor.mainHand then
		local min, max = calcDamage(actor.mainSkill, output, {}, breakdown, "Physical", 0)
		output.MainHandMin = min
		output.MainHandMax = max
		output.MainHandAverage = (min + max) / 2
	end

	return output, breakdown
end

-- Export module functions
return {
	calcDamage = calcDamage,
	calcAilmentDamage = calcAilmentDamage,
	mergeModDB = mergeModDB,
	skillHasFlag = skillHasFlag,
	buildOutput = buildOutput,
	dmgTypeList = dmgTypeList,
	dmgTypeFlags = dmgTypeFlags,
	ailmentData = ailmentData,
}
