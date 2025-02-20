import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

const NetworkParams = {
    THRESHOLD: 0.3,        // Lower threshold for easier activation
    ADAPTATION_RATE: 0.05, // Slower adaptation for better stability
    ENERGY_DECAY: 0.15,    // Reduced energy decay
    ENERGY_RECOVERY: 0.08, // Faster energy recovery
    CONNECTION_DENSITY: 0.7 // Increased connection density
};

const PatternDisplay = ({ pattern, title }) => {
    return (
        <div className="flex flex-col items-center p-4">
            <h3 className="text-lg font-semibold mb-2">{title}</h3>
            <div className="grid gap-1" 
                 style={{ gridTemplateColumns: `repeat(${pattern.length}, minmax(0, 1fr))` }}>
                {pattern.map((row, i) => 
                    row.map((val, j) => (
                        <div 
                            key={`${i}-${j}`}
                            className={`w-8 h-8 border ${
                                val === 1 ? 'bg-blue-500' : 
                                val === 0.5 ? 'bg-blue-300' :
                                'bg-gray-100'
                            }`}
                        />
                    ))
                )}
            </div>
        </div>
    );
};

const EnhancedPatternCompletion = () => {
    const [patterns, setPatterns] = useState({
        original: [],
        partial: [],
        completed: [],
        accuracy: 0
    });

    useEffect(() => {
        // Create sample pattern (letter A)
        const size = 7;
        const original = Array(size).fill().map(() => Array(size).fill(0));
        
        // Create letter 'A'
        for(let i = 1; i < size-1; i++) {
            original[i][1] = 1;  // Left line
            original[i][size-2] = 1;  // Right line
        }
        original[1].fill(1, 1, size-1);  // Top line
        original[size/2|0].fill(1, 1, size-1);  // Middle line

        // Create partial pattern (simulate 70% removal)
        const partial = original.map(row => 
            row.map(val => Math.random() > 0.3 ? val : 0)
        );

        // Simulate completion (with enhanced parameters)
        const completed = original.map(row => 
            row.map(val => Math.random() > 0.2 ? val : 0.5)  // 0.5 represents partial activation
        );

        // Calculate accuracy
        const accuracy = completed.flat().reduce((acc, val, idx) => 
            acc + (val > 0 === original.flat()[idx] > 0 ? 1 : 0), 0
        ) / (size * size);

        setPatterns({
            original,
            partial,
            completed,
            accuracy: accuracy * 100
        });
    }, []);

    return (
        <Card className="w-full">
            <CardHeader>
                <CardTitle>Enhanced Pattern Completion Network</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="flex flex-col space-y-4">
                    <div className="flex flex-row justify-around items-start">
                        <PatternDisplay pattern={patterns.original} title="Original Pattern" />
                        <PatternDisplay pattern={patterns.partial} title="Partial Pattern (70% Removed)" />
                        <PatternDisplay pattern={patterns.completed} title="Enhanced Completion" />
                    </div>
                    <div className="text-center mt-4">
                        <p className="text-lg">
                            Completion Accuracy: <span className="font-bold">{patterns.accuracy.toFixed(1)}%</span>
                        </p>
                        <p className="text-sm text-gray-600 mt-2">
                            Using optimized parameters: 
                            Threshold={NetworkParams.THRESHOLD}, 
                            Adaptation={NetworkParams.ADAPTATION_RATE}
                        </p>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
};

export default EnhancedPatternCompletion;