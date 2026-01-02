-- tests/test_export_dataset.lua
local lu = require('tests.luaunit')

-- --- MOCKS ---
-- Mock BizHawk globals: emu, joypad, client, movie, console, os (partially)
emu = {
    getsystemid = function() return "SNES" end -- Default for loading
}
joypad = {}
client = {}
movie = {}
console = {}
_G.joypad = joypad
_G.emu = emu

console.log = function(...) end -- silence logs

-- Stub os.execute to check directory creation
local executed_commands = {}
local original_os_execute = os.execute
os.execute = function(cmd)
    table.insert(executed_commands, cmd)
    return true
end

-- Mock io.open and file writes
local mock_fs = {}
local mock_files = {}

local MockFile = {}
MockFile.__index = MockFile

function MockFile.new(path)
    local self = setmetatable({}, MockFile)
    self.path = path
    self.content = ""
    return self
end

function MockFile:write(data)
    self.content = self.content .. data
end

function MockFile:close()
    -- no-op
end

io.open = function(path, mode)
    if mode == "w" then
        local file = MockFile.new(path)
        mock_files[path] = file
        return file
    end
    return nil, "File not found (mock)"
end

-- --- LOAD SCRIPT UNDER TEST ---
TEST_MODE = true -- Flag to prevent main loop execution
local script_path = "export_dataset.lua"

local status, result = pcall(dofile, script_path)

if not status then
    print("\n[FATAL] Failed to load script: " .. tostring(result))
    os.exit(1)
end

local app = result

if type(app) ~= "table" then
    print("\n[FATAL] Script did not return export table. Got: " .. type(app))
    os.exit(1)
end

-- Clear TEST_MODE so LuaUnit doesn't try to run it as a test suite
TEST_MODE = nil

-- --- TESTS ---

TestExportDataset = {}

    function TestExportDataset:test_export_required_functions()
        lu.assertEquals(type(app.get_nitrogen_input), "function")
        lu.assertEquals(type(app.bool_to_int), "function")
    end

    function TestExportDataset:test_bool_to_int()
        lu.assertEquals(app.bool_to_int(true), 1)
        lu.assertEquals(app.bool_to_int(false), 0)
        lu.assertEquals(app.bool_to_int(nil), 0)
    end

    function TestExportDataset:test_snes_mapping()
        -- Setup SNES (mock logic for joypad only)
        joypad.get = function() 
            return {
                B=true, A=false, Y=true, X=false,
                L=true, R=false,
                Start=true, Select=false,
                Up=true, Down=false, Left=false, Right=true
            }
        end
        
        local result = app.get_nitrogen_input("SNES")
        -- Expected: S,E,W,N, L,R, LT,RT, St,Bk, Up,Dn,Lf,Rt, X,Y
        -- SNES: S=B(1), E=A(0), W=Y(1), N=X(0)
        local expected = "1,0,1,0,1,0,0,0,1,0,1,0,0,1,0.0,0.0"
        lu.assertEquals(result, expected)
    end

    function TestExportDataset:test_nes_mapping()
        -- Setup NES
        joypad.get = function() 
            return {
                B=true, A=false, -- B=South, A=East
                Start=true, Select=true,
                Up=false, Down=false, Left=true, Right=false
            }
        end
        
        local result = app.get_nitrogen_input("NES")
        -- S=1, E=0, W=0, N=0, St=1, Bk=1, Lf=1
        local expected = "1,0,0,0,0,0,0,0,1,1,0,0,1,0,0.0,0.0"
        lu.assertEquals(result, expected)
    end

    function TestExportDataset:test_generic_mapping()
        joypad.get = function()
            return {
                A=true, B=false, C=true, -- S=A, E=B, W=C
                Start=true, Mode=false,
                Up=true
            }
        end
        
        local result = app.get_nitrogen_input("GENESIS")
        -- S=1, E=0, W=1, St=1, Up=1
        local expected = "1,0,1,0,0,0,0,0,1,0,1,0,0,0,0.0,0.0"
        lu.assertEquals(result, expected)
    end

os.exit(lu.LuaUnit.run())
