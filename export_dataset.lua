-- export_dataset.lua
-- Script to export frames and input actions from a .bk2 movie for NitroGen training
-- Based on BizHawk Lua API and NitroGen requirements

-- CONFIGURATION
-- =============================================================================
local output_base_dir = "nitrogen_dataset" -- Default relative path
-- You can specify an absolute path if preferred, e.g.:
-- local output_base_dir = "/home/user/nitrogen_dataset/"
local frames_subdir = "frames"
local csv_filename = "actions.csv"

-- UTILITIES
-- =============================================================================

---Check if the operating system is Windows
---@return boolean
local function is_windows()
    return package.config:sub(1,1) == "\\"
end

---Create a directory if it doesn't exist
---@param path string
local function create_directory(path)
    if is_windows() then
        os.execute("mkdir \"" .. path .. "\"")
    else
        os.execute("mkdir -p \"" .. path .. "\"")
    end
end

---Convert boolean to integer (1 or 0)
---@param val boolean|nil
---@return integer
local function bool_to_int(val)
    return val and 1 or 0
end

-- MAIN LOGIC
-- =============================================================================

-- 1. Setup paths
local output_dir = output_base_dir
if not output_dir:match("^/") and not output_dir:match("^%a:") then
     -- If relative, try to make it absolute or just use clear struct
    -- BizHawk usually runs from its own dir, so relative paths land there.
    -- We'll keep it simple.
end

-- Ensure trailing slash
if output_dir:sub(-1) ~= "/" and output_dir:sub(-1) ~= "\\" then
    output_dir = output_dir .. "/"
end

local frames_full_path = output_dir .. frames_subdir .. "/"
local csv_full_path = output_dir .. csv_filename

-- 2. Detect System
local CONSOLE_TYPE = emu.getsystemid()
console.log("System detected: " .. CONSOLE_TYPE)

if CONSOLE_TYPE ~= "NES" and CONSOLE_TYPE ~= "SNES" then
    console.log("Warning: System " .. CONSOLE_TYPE .. " logic is not strictly defined, defaulting to generic mapping.")
end

-- 3. Initialize Output
console.log("Creating output directory: " .. frames_full_path)
create_directory(frames_full_path)

local file, err = io.open(csv_full_path, "w")
if not file then
    console.log("Error opening CSV file for writing: " .. tostring(err))
    return
end

-- Write CSV Header (NitroGen format)
file:write("frame,south,east,west,north,left_shoulder,right_shoulder,left_trigger,right_trigger,start,back,dpad_up,dpad_down,dpad_left,dpad_right,stick_x,stick_y\n")

-- 3.1 Write Configuration for Python Converter
local config_path = output_dir .. "dataset_config.json"
local config_file = io.open(config_path, "w")
if config_file then
    local resize_mode = "pad" -- Default
    if CONSOLE_TYPE == "NES" then
        resize_mode = "crop"
    elseif CONSOLE_TYPE == "SNES" then
        resize_mode = "pad"
    end
    
    -- Simple JSON manually to avoid dependencies
    config_file:write(string.format('{\n    "resize_mode": "%s",\n    "console_type": "%s"\n}', resize_mode, CONSOLE_TYPE))
    config_file:close()
    console.log("Config saved: " .. resize_mode)
else
    console.log("Error writing config file.")
end

-- 4. Input Mapping Logic
---Get current frame input and format it for NitroGen CSV
---@param system_id string
---@return string
local function get_nitrogen_input(system_id)
    -- CHANGE 1: Read input from the movie for the current frame, not from the gamepad
    local pad = movie.getinput(emu.framecount())
    
    local south, east, west, north = 0, 0, 0, 0
    local l_sh, r_sh, l_tr, r_tr = 0, 0, 0, 0
    local start, back = 0, 0
    local up, down, left, right = 0, 0, 0, 0
    
    -- CHANGE 2: Add "P1 " prefix to all buttons
    
    if system_id == "SNES" then
        -- SNES Layout:
        south = bool_to_int(pad["P1 B"])
        east  = bool_to_int(pad["P1 A"])
        west  = bool_to_int(pad["P1 Y"])
        north = bool_to_int(pad["P1 X"])
        
        l_sh  = bool_to_int(pad["P1 L"])
        r_sh  = bool_to_int(pad["P1 R"])
        
        start = bool_to_int(pad["P1 Start"])
        back  = bool_to_int(pad["P1 Select"])
        
        up    = bool_to_int(pad["P1 Up"])
        down  = bool_to_int(pad["P1 Down"])
        left  = bool_to_int(pad["P1 Left"])
        right = bool_to_int(pad["P1 Right"])
        
    elseif system_id == "NES" then
        -- NES Layout:
        -- B -> South, A -> East
        south = bool_to_int(pad["P1 B"]) 
        east  = bool_to_int(pad["P1 A"]) 
        
        start = bool_to_int(pad["P1 Start"])
        back  = bool_to_int(pad["P1 Select"])
        
        up    = bool_to_int(pad["P1 Up"])
        down  = bool_to_int(pad["P1 Down"])
        left  = bool_to_int(pad["P1 Left"])
        right = bool_to_int(pad["P1 Right"])
        
    else
        -- GENERIC / FALLBACK
        south = bool_to_int(pad["P1 A"]) 
        east  = bool_to_int(pad["P1 B"])
        west  = bool_to_int(pad["P1 C"]) 
        
        start = bool_to_int(pad["P1 Start"] or pad["P1 S"])
        back  = bool_to_int(pad["P1 Select"] or pad["P1 Mode"])
        
        up    = bool_to_int(pad["P1 Up"])
        down  = bool_to_int(pad["P1 Down"])
        left  = bool_to_int(pad["P1 Left"])
        right = bool_to_int(pad["P1 Right"])
    end

    -- Stick X/Y logic remains 0.0 for NES/SNES
    return string.format("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,0.0,0.0",
        south, east, west, north,
        l_sh, r_sh, l_tr, r_tr,
        start, back,
        up, down, left, right
    )
end

-- 5. Main Loop
if not TEST_MODE then
    console.log("Starting export...")
    console.log("Please ensure a movie (.bk2) is currently playing.")

    while true do
        local frame = emu.framecount()
        
        -- Capture Screenshot
        -- Note: client.screenshot includes OSD if visible. 
        -- Ideally use gui.clearGraphics() or configuration to hide HUD if needed.
        client.screenshot(frames_full_path .. string.format("frame_%06d.png", frame))
        
        -- Capture Input
        local input_data = get_nitrogen_input(CONSOLE_TYPE)
        file:write(frame .. "," .. input_data .. "\n")
        
        -- Advance Frame
        emu.frameadvance()
        
        -- Check for Movie End
        if movie.isloaded() and movie.mode() == "FINISHED" then
            file:close()
            console.log("Export complete!")
            console.log("Data saved to: " .. output_dir)
            console.log("Now run the Python converter script.")
            break
        end
    end
end

-- Export functions for testing
return {
    get_nitrogen_input = get_nitrogen_input,
    bool_to_int = bool_to_int,
    output_dir = output_dir,
    frames_full_path = frames_full_path,
    csv_full_path = csv_full_path
}

